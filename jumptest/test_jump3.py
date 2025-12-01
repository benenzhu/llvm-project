#!/usr/bin/env python3
"""Test template context for jump3.cpp - D::rows hover should show 64."""

import subprocess
import json
import os
import time

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/jumptest/jump3.cpp"

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
    
    print("Waiting for indexing (10s)...")
    time.sleep(10)
    
    # Step 1: Goto definition from mma_ABt at line 61
    # Line 61: "  mma_ABt(C_accum[0], tiles[1]);"
    # mma_ABt starts at column 2
    print("\n=== Step 1: Goto definition from mma_ABt at line 61 ===")
    def_id = send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 60, "character": 3}  # 0-based: line 60, char 3 = middle of mma_ABt
    })
    resp = wait_for(def_id)
    if resp and resp.get('result'):
        result = resp['result']
        if isinstance(result, list) and len(result) > 0:
            loc = result[0]
            line = loc.get('range', loc.get('targetRange', {})).get('start', {}).get('line', -1)
            print(f"  Jumped to line {line+1 if line >= 0 else '?'}")
        else:
            print(f"  Result: {result}")
    else:
        print(f"  No result: {resp}")
    
    time.sleep(2)
    
    # Step 2: Hover on D::rows at line 37
    # Line 37: "  static_assert(D::rows == A::rows);"
    # D::rows starts around column 17 (D at 17, :: at 18-19, rows at 20-23)
    print("\n=== Step 2: Hover on 'rows' in D::rows at line 37 ===")
    for col in [20, 21, 22]:
        hover_id = send("textDocument/hover", {
            "textDocument": {"uri": f"file://{TEST_FILE}"},
            "position": {"line": 36, "character": col}
        })
        resp = wait_for(hover_id, timeout=5)
        if resp and resp.get('result'):
            contents = resp['result'].get('contents', {})
            value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
            print(f"  Column {col}: {value[:200]}...")
            if '64' in value:
                print("  ✓ Found 64!")
                break
        else:
            print(f"  Column {col}: No result")
    
    # Step 3: Goto definition from D::rows
    print("\n=== Step 3: Goto definition from D::rows ===")
    def_id = send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 36, "character": 20}
    })
    resp = wait_for(def_id)
    if resp and resp.get('result'):
        result = resp['result']
        print(f"  Definition result: {json.dumps(result, indent=2)[:500]}")
    else:
        print(f"  No result: {resp}")
    
    # Step 4: Hover on template parameter D at line 36
    # Line 37: "  static_assert(D::rows == A::rows);"
    # D is at column 17
    print("\n=== Step 4: Hover on template parameter 'D' at line 37 ===")
    hover_id = send("textDocument/hover", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 36, "character": 17}
    })
    resp = wait_for(hover_id, timeout=5)
    if resp and resp.get('result'):
        contents = resp['result'].get('contents', {})
        value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
        print(f"  Hover result:\n{value[:500]}")
        if 'rt<float' in value or 'rt_fl' in value:
            print("  ✓ Shows instantiated type!")
    else:
        print(f"  No result: {resp}")
    
    time.sleep(1)
    
    # Step 5: Hover on _rows at line 50 (after Step 3 jumped there)
    # Line 50: "    static constexpr int rows = _rows;"
    print("\n=== Step 5: Hover on '_rows' at line 50 ===")
    for col in [36, 37, 38, 39, 40]:
        hover_id = send("textDocument/hover", {
            "textDocument": {"uri": f"file://{TEST_FILE}"},
            "position": {"line": 49, "character": col}
        })
        resp = wait_for(hover_id, timeout=5)
        if resp and resp.get('result'):
            contents = resp['result'].get('contents', {})
            value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
            print(f"  Column {col}: {value[:300]}")
            if '64' in value:
                print("  ✓ Shows value 64!")
                break
        else:
            print(f"  Column {col}: No result")
    
    time.sleep(1)
    
    # Step 6: Jump back from rows at line 50 to somewhere in function template
    # Goto definition from "rows" at line 50 (the variable name, not _rows)
    print("\n=== Step 6: Goto definition from 'rows' at line 50 ===")
    def_id = send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 49, "character": 27}  # 'rows' in "static constexpr int rows"
    })
    resp = wait_for(def_id)
    if resp and resp.get('result'):
        result = resp['result']
        if isinstance(result, list) and len(result) > 0:
            loc = result[0]
            line = loc.get('range', {}).get('start', {}).get('line', -1)
            print(f"  Jumped to line {line+1 if line >= 0 else '?'}")
    
    time.sleep(1)
    
    # Step 7: After jumping back, hover on D::rows again to see if context is restored
    print("\n=== Step 7: Hover on D::rows at line 37 (after jumping back) ===")
    for col in [20, 21, 22]:
        hover_id = send("textDocument/hover", {
            "textDocument": {"uri": f"file://{TEST_FILE}"},
            "position": {"line": 36, "character": col}
        })
        resp = wait_for(hover_id, timeout=5)
        if resp and resp.get('result'):
            contents = resp['result'].get('contents', {})
            value = contents.get('value', '') if isinstance(contents, dict) else str(contents)
            print(f"  Column {col}: {value[:200]}...")
            if '64' in value:
                print("  ✓ Context restored! Shows 64!")
                break
        else:
            print(f"  Column {col}: No result")
    
    proc.terminate()
    
    # Print stderr
    stderr = proc.stderr.read().decode()
    print("\n=== Relevant logs ===")
    for line in stderr.split('\n'):
        if any(k in line for k in ["locateAST", "Candidate", "getHover", "Template", 
                                    "Instantiation", "Found"]):
            print(line)

if __name__ == "__main__":
    main()

