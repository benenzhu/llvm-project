#!/usr/bin/env python3
"""Simple test for goto definition and hover."""

import subprocess
import json
import os
import time

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/jumptest/jump2.cpp"

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
    
    req_id = 0
    
    def send(method, params, is_notification=False):
        nonlocal req_id
        request = {"jsonrpc": "2.0", "method": method, "params": params}
        if not is_notification:
            req_id += 1
            request["id"] = req_id
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        proc.stdin.write(message.encode())
        proc.stdin.flush()
        return req_id if not is_notification else None
    
    def recv(timeout=30):
        import select
        header_data = b""
        start = time.time()
        while time.time() - start < timeout:
            if select.select([proc.stdout], [], [], 0.1)[0]:
                char = proc.stdout.read(1)
                if not char:
                    return None
                header_data += char
                if header_data.endswith(b"\r\n\r\n"):
                    break
        else:
            return None
        
        headers = {}
        for line in header_data.decode().split('\r\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                headers[k.strip()] = v.strip()
        
        content_length = int(headers.get('Content-Length', 0))
        if content_length > 0:
            content = proc.stdout.read(content_length).decode()
            return json.loads(content)
        return None
    
    def wait_for(expected_id, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            msg = recv(timeout=1)
            if msg and msg.get('id') == expected_id:
                return msg
        return None
    
    # Initialize
    print("Initializing...")
    init_id = send("initialize", {
        "processId": os.getpid(),
        "rootUri": f"file:///A/clangd-dev/jumptest",
        "capabilities": {}
    })
    resp = wait_for(init_id)
    print(f"  Init response: {'OK' if resp else 'FAIL'}")
    
    send("initialized", {}, is_notification=True)
    
    # Open file
    print(f"Opening {TEST_FILE}...")
    send("textDocument/didOpen", {
        "textDocument": {
            "uri": f"file://{TEST_FILE}",
            "languageId": "cpp",
            "version": 1,
            "text": file_content
        }
    }, is_notification=True)
    
    print("Waiting for indexing (10s)...")
    time.sleep(10)
    
    # Goto definition from mma_AB(a) at line 18
    print("\nGoto definition from mma_AB(a) at line 18, char 5...")
    def_id = send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 17, "character": 5}
    })
    resp = wait_for(def_id)
    print(f"  Response: {json.dumps(resp, indent=2) if resp else 'TIMEOUT'}")
    
    time.sleep(1)
    
    # Hover on 'now' at line 13
    print("\nHover on 'now' at line 13, char 15...")
    hover_id = send("textDocument/hover", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 12, "character": 15}
    })
    resp = wait_for(hover_id)
    if resp and resp.get('result'):
        contents = resp['result'].get('contents', {})
        value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
        print(f"  Hover result:\n{value[:500]}")
    else:
        print(f"  Response: {resp}")
    
    proc.terminate()
    
    # Print stderr
    stderr = proc.stderr.read().decode()
    print("\n=== Relevant logs ===")
    for line in stderr.split('\n'):
        if any(k in line for k in ["locateAST", "Candidate", "getHover", "Template"]):
            print(line)

if __name__ == "__main__":
    main()
