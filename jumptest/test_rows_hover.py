#!/usr/bin/env python3
"""
Test hover on 'rows' in 'a.rows' to verify it shows the value.
"""

import subprocess
import json
import sys
import os
import time
import threading

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/jumptest/jump2.cpp"

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
    import select
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
    
    for line in header_data.decode().split('\r\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
    
    content_length = int(headers.get('Content-Length', 0))
    if content_length > 0:
        content = proc.stdout.read(content_length).decode()
        return json.loads(content)
    return None

def wait_for_response(proc, expected_id, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        msg = read_message(proc, timeout=5)
        if msg is None:
            continue
        if msg.get('id') == expected_id:
            return msg
        if 'method' in msg:
            print(f"  [notification] {msg['method']}")
    return None

def stderr_reader(proc, output_list):
    while True:
        line = proc.stderr.readline()
        if not line:
            break
        output_list.append(line.decode())

def main():
    with open(TEST_FILE, 'r') as f:
        file_content = f.read()
    
    proc = subprocess.Popen(
        [CLANGD_PATH, "--log=verbose", "-j=4"],
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
        "rootUri": "file:///A/clangd-dev/jumptest",
        "capabilities": {"textDocument": {"hover": {}}}
    }, 1)
    
    resp = wait_for_response(proc, 1)
    print(f"Initialize: {'OK' if resp else 'FAILED'}")
    
    send_request(proc, "initialized", {}, None)
    
    # Open file
    send_request(proc, "textDocument/didOpen", {
        "textDocument": {
            "uri": f"file://{TEST_FILE}",
            "languageId": "cpp",
            "version": 1,
            "text": file_content
        }
    }, None)
    
    print("File opened, waiting for indexing...")
    time.sleep(8)  # Wait longer for indexing
    
    # Print any diagnostics
    for line in stderr_output:
        if "diagnostic" in line.lower() or "error" in line.lower():
            print(f"  DIAG: {line.rstrip()}")
    
    # Step 1: goto definition from mma_AB(a) at line 18 (0-based: 17)
    print("\n=== Step 1: Goto definition from mma_AB(a) at line 18 ===")
    # Line 18: "    mma_AB(a);"  -- mma_AB starts at character 4
    send_request(proc, "textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 17, "character": 5}  # middle of 'mma_AB' call
    }, 2)
    
    resp = wait_for_response(proc, 2, timeout=30)
    if resp and resp.get('result'):
        result = resp['result']
        loc = result[0] if isinstance(result, list) else result
        target_line = loc.get('range', loc.get('targetRange', {})).get('start', {}).get('line', -1)
        print(f"  Jumped to: line {target_line+1 if target_line >= 0 else '?'}")
        print(f"  Full response: {result}")
    else:
        print(f"  No result or error: {resp}")
    
    time.sleep(2)
    
    # Step 2: Hover on 'now' at line 13 (0-based: 12)
    print("\n=== Step 2: Hover on 'now' at line 13 ===")
    send_request(proc, "textDocument/hover", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 12, "character": 15}  # 'now' variable
    }, 3)
    
    resp = wait_for_response(proc, 3)
    if resp and resp.get('result'):
        result = resp['result']
        contents = result.get('contents', {})
        value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
        print(f"  Hover result: {value[:200]}...")
        if 'Value' in value or 'value' in value.lower():
            print("  ✓ Shows value!")
        else:
            print("  ✗ No value shown")
    
    # Step 3: Hover on 'rows' in 'a.rows' at line 13 (0-based: 12)
    # Line: "    const auto now = a.rows;"
    # Positions: a=20, .=21, r=22, o=23, w=24, s=25
    print("\n=== Step 3: Hover on 'rows' in 'a.rows' at line 13 ===")
    for col in [22, 23, 24, 25]:
        print(f"  Trying column {col}...")
        send_request(proc, "textDocument/hover", {
            "textDocument": {"uri": f"file://{TEST_FILE}"},
            "position": {"line": 12, "character": col}
        }, 4 + col)
        
        resp = wait_for_response(proc, 4 + col, timeout=5)
        if resp and resp.get('result'):
            result = resp['result']
            contents = result.get('contents', {})
            value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
            print(f"    Hover result (column {col}):")
            print(f"    {value[:300]}")
            if 'Value' in value or 'value' in value.lower():
                print("    ✓ Shows value!")
            break
        else:
            print(f"    No hover result at column {col}")
    
    proc.terminate()
    proc.wait()
    
    # Print relevant stderr logs
    print("\n=== Relevant clangd logs ===")
    for line in stderr_output:
        if any(k in line for k in ["getHover", "TemplateCtx", "locateAST", 
                                    "Instantiation", "FunctionDecl", "AddResult"]):
            print(line.rstrip())

if __name__ == "__main__":
    main()

