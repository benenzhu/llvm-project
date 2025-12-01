#!/usr/bin/env python3
"""Simple test for load jump filtering."""

import subprocess
import json
import sys
import time
import threading

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/hip-test/read.cpp"

def reader_thread(proc, results):
    """Read messages from clangd."""
    while True:
        try:
            header = b""
            while not header.endswith(b"\r\n\r\n"):
                b = proc.stdout.read(1)
                if not b:
                    return
                header += b
            
            content_length = 0
            for line in header.decode().split('\r\n'):
                if line.startswith('Content-Length:'):
                    content_length = int(line.split(':')[1].strip())
            
            if content_length > 0:
                body = proc.stdout.read(content_length).decode()
                msg = json.loads(body)
                results.append(msg)
        except Exception as e:
            print(f"Reader error: {e}")
            return

def main():
    print(f"Loading {TEST_FILE}...")
    with open(TEST_FILE, 'r') as f:
        content = f.read()
    
    print("Starting clangd...")
    proc = subprocess.Popen(
        [CLANGD_PATH, "--log=verbose", "-j=4"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    results = []
    reader = threading.Thread(target=reader_thread, args=(proc, results))
    reader.daemon = True
    reader.start()
    
    def send(method, params, req_id=None):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if req_id is not None:
            msg["id"] = req_id
        body = json.dumps(msg)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        proc.stdin.write(header.encode() + body.encode())
        proc.stdin.flush()
    
    def wait_response(req_id, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            for r in results:
                if r.get('id') == req_id:
                    return r
            time.sleep(0.1)
        return None
    
    # Initialize
    print("Initializing...")
    send("initialize", {
        "processId": 1,
        "rootUri": "file:///A/clangd-dev/hip-test",
        "capabilities": {}
    }, 1)
    
    resp = wait_response(1, timeout=30)
    print(f"  Init: {'OK' if resp else 'TIMEOUT'}")
    if not resp:
        proc.terminate()
        return
    
    send("initialized", {})
    
    # Open file
    print("Opening file...")
    send("textDocument/didOpen", {
        "textDocument": {
            "uri": f"file://{TEST_FILE}",
            "languageId": "cpp",
            "version": 1,
            "text": content
        }
    })
    
    print("Waiting for indexing (20s)...")
    time.sleep(20)
    
    # Test jump from line 7814
    # Line 7814: "    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);"
    print("\n=== Testing jump from kittens::load at line 7814 ===")
    send("textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": 7813, "character": 14}
    }, 2)
    
    resp = wait_response(2, timeout=30)
    if resp:
        result = resp.get('result', [])
        if isinstance(result, list):
            print(f"  Got {len(result)} result(s)")
            if len(result) == 1:
                print("  ✓ SUCCESS: Only 1 result!")
            else:
                print(f"  ✗ Expected 1, got {len(result)}")
            for i, loc in enumerate(result[:5]):
                line = loc.get('range', {}).get('start', {}).get('line', -1)
                uri = loc.get('uri', '').split('/')[-1]
                print(f"    [{i}] line {line+1} in {uri}")
    else:
        print("  TIMEOUT")
    
    proc.terminate()
    
    # Print relevant logs
    stderr = proc.stderr.read().decode()
    print("\n=== Logs ===")
    for line in stderr.split('\n')[-50:]:
        if any(k in line for k in ["Candidates", "filtering", "specialization", "Found"]):
            print(line)

if __name__ == "__main__":
    main()

