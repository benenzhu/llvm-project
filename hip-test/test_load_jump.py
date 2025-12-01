#!/usr/bin/env python3
"""
Test goto definition for load function at line 9102 in read.cpp
"""

import subprocess
import json
import sys
import os
import time
import threading

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/hip-test/read.cpp"

def send_request(proc, method, params, req_id):
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    }
    if req_id is not None:
        request["id"] = req_id
    content = json.dumps(request)
    message = f"Content-Length: {len(content)}\r\n\r\n{content}"
    proc.stdin.write(message.encode())
    proc.stdin.flush()

def read_message(proc, timeout=60):
    """Read a single message from clangd"""
    import select
    
    # Read headers
    headers = {}
    header_data = b""
    while True:
        if select.select([proc.stdout], [], [], timeout)[0]:
            char = proc.stdout.read(1)
            if not char:
                return None
            header_data += char
            if header_data.endswith(b"\r\n\r\n"):
                break
        else:
            return None
    
    # Parse headers
    for line in header_data.decode().split('\r\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
    
    # Read content
    content_length = int(headers.get('Content-Length', 0))
    if content_length > 0:
        content = proc.stdout.read(content_length).decode()
        return json.loads(content)
    return None

def wait_for_response(proc, expected_id, timeout=120):
    """Wait for a specific response, handling notifications"""
    start = time.time()
    while time.time() - start < timeout:
        msg = read_message(proc, timeout=5)
        if msg is None:
            continue
        if msg.get('id') == expected_id:
            return msg
        # Print notifications for debugging
        if 'method' in msg:
            print(f"  [notification] {msg['method']}")
    return None

def stderr_reader(proc, output_list):
    """Read stderr in background"""
    while True:
        line = proc.stderr.readline()
        if not line:
            break
        output_list.append(line.decode())

def main():
    with open(TEST_FILE, 'r') as f:
        file_content = f.read()
    
    proc = subprocess.Popen(
        [CLANGD_PATH, "--log=verbose", "-j=8"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    
    stderr_output = []
    stderr_thread = threading.Thread(target=stderr_reader, args=(proc, stderr_output))
    stderr_thread.daemon = True
    stderr_thread.start()
    
    print("Started clangd")
    
    # Initialize
    send_request(proc, "initialize", {
        "processId": os.getpid(),
        "rootUri": "file:///A/clangd-dev/hip-test",
        "capabilities": {"textDocument": {"hover": {}}}
    }, 1)
    
    resp = wait_for_response(proc, 1)
    print(f"Initialize: {'OK' if resp else 'FAILED'}")
    
    send_request(proc, "initialized", {}, None)
    
    # Open file
    send_request(proc, "textDocument/didOpen", {
        "textDocument": {
            "uri": f"file://{TEST_FILE}",
            "languageId": "hip",
            "version": 1,
            "text": file_content
        }
    }, None)
    
    print("File opened, waiting for indexing (30s)...")
    time.sleep(30)
    
    # Test goto definition at line 9102 (0-based: 9101), column 8 (the 'load' call)
    line = 9101
    char = 8
    
    print(f"\nTesting goto definition at line {line+1}, char {char}")
    send_request(proc, "textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": line, "character": char}
    }, 2)
    
    resp = wait_for_response(proc, 2, timeout=60)
    print(f"\nGoto definition response:")
    if resp:
        if 'result' in resp:
            result = resp['result']
            if result:
                for loc in (result if isinstance(result, list) else [result]):
                    uri = loc.get('uri', loc.get('targetUri', 'unknown'))
                    range_info = loc.get('range', loc.get('targetRange', {}))
                    start = range_info.get('start', {})
                    print(f"  -> {uri}")
                    print(f"     line {start.get('line', 0)+1}, char {start.get('character', 0)}")
            else:
                print("  No results")
        elif 'error' in resp:
            print(f"  Error: {resp['error']}")
    else:
        print("  No response (timeout)")
    
    # Print relevant stderr logs
    print("\n--- Relevant debug logs ---")
    for line in stderr_output:
        if 'load' in line.lower() or 'TemplateInstantiation' in line or '9102' in line or '9101' in line:
            print(line.rstrip())
    
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()

