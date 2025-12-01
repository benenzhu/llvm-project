#!/usr/bin/env python3
"""
Test chained jump from read.cpp:
1. Jump from line 9064 G::load(...) to line 7814
2. Jump from line 7814 kittens::load<...> - should return only ONE result
"""

import subprocess
import json
import os
import time

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/hip-test/read.cpp"

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
    
    def recv(timeout=60):
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
    
    def wait_for(expected_id, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            msg = recv(timeout=2)
            if msg is None:
                continue
            if msg.get('id') == expected_id:
                return msg
            if 'method' in msg:
                method = msg['method']
                if method != 'textDocument/publishDiagnostics':
                    print(f"  [notification] {method}")
        print(f"  Timeout waiting for response {expected_id}")
        return None
    
    # Initialize
    print("Initializing...")
    init_id = send("initialize", {
        "processId": os.getpid(),
        "rootUri": f"file:///A/clangd-dev/hip-test",
        "capabilities": {}
    })
    resp = wait_for(init_id)
    print(f"  Init: {'OK' if resp else 'FAIL'}")
    
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
    
    print("Waiting for indexing (15s)...")
    time.sleep(15)
    
    # Step 1: Jump from line 9064 G::load to wherever it goes
    # Line 9064: "    G::load(As, g.a, {0, 0, row, 0});"
    # 'load' starts around column 7
    print("\n=== Step 1: Goto definition from G::load at line 9064 ===")
    def_id = send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 9063, "character": 7}  # 0-based line 9063
    })
    resp = wait_for(def_id)
    first_jump_line = None
    if resp and resp.get('result'):
        result = resp['result']
        if isinstance(result, list) and len(result) > 0:
            print(f"  Got {len(result)} result(s)")
            for i, loc in enumerate(result[:3]):
                line = loc.get('range', {}).get('start', {}).get('line', -1)
                uri = loc.get('uri', '')
                print(f"    [{i}] line {line+1} in {uri.split('/')[-1]}")
                if i == 0:
                    first_jump_line = line
        else:
            print(f"  Result: {result}")
    else:
        print(f"  No result: {resp}")
    
    time.sleep(2)
    
    # Step 2: Jump from line 7814 kittens::load
    # Line 7814: "    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);"
    # 'load' is at around column 14
    print("\n=== Step 2: Goto definition from kittens::load at line 7814 ===")
    def_id = send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 7813, "character": 14}  # 0-based line 7813
    })
    resp = wait_for(def_id)
    if resp and resp.get('result'):
        result = resp['result']
        if isinstance(result, list):
            print(f"  Got {len(result)} result(s)")
            if len(result) == 1:
                print("  ✓ SUCCESS: Only 1 result!")
            else:
                print(f"  ✗ FAIL: Expected 1 result, got {len(result)}")
            for i, loc in enumerate(result[:5]):
                line = loc.get('range', {}).get('start', {}).get('line', -1)
                uri = loc.get('uri', '')
                print(f"    [{i}] line {line+1} in {uri.split('/')[-1]}")
        else:
            print(f"  Result: {result}")
    else:
        print(f"  No result: {resp}")
    
    proc.terminate()
    
    # Print stderr logs
    stderr = proc.stderr.read().decode()
    print("\n=== Relevant logs ===")
    for line in stderr.split('\n'):
        if any(k in line for k in ["locateAST", "Candidate", "Template", 
                                    "Instantiation", "Found", "filtering",
                                    "specialization"]):
            print(line)

if __name__ == "__main__":
    main()

