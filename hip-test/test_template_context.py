#!/usr/bin/env python3
"""
Test script to diagnose template instantiation context issues in read.cpp
"""

import subprocess
import json
import sys
import os

# Path to clangd
CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
TEST_FILE = "/A/clangd-dev/hip-test/read.cpp"

def send_request(proc, method, params, req_id):
    """Send a JSON-RPC request to clangd"""
    request = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params
    }
    content = json.dumps(request)
    message = f"Content-Length: {len(content)}\r\n\r\n{content}"
    proc.stdin.write(message.encode())
    proc.stdin.flush()

def read_response(proc):
    """Read a JSON-RPC response from clangd"""
    # Read headers
    headers = {}
    while True:
        line = proc.stdout.readline().decode().strip()
        if not line:
            break
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
    
    # Read content
    content_length = int(headers.get('Content-Length', 0))
    if content_length > 0:
        content = proc.stdout.read(content_length).decode()
        return json.loads(content)
    return None

def wait_for_response(proc, expected_id, timeout=60):
    """Wait for a specific response"""
    import select
    while True:
        if select.select([proc.stdout], [], [], timeout)[0]:
            resp = read_response(proc)
            if resp and resp.get('id') == expected_id:
                return resp
        else:
            return None

def main():
    # Read the file content
    with open(TEST_FILE, 'r') as f:
        file_content = f.read()
    
    # Start clangd with verbose logging
    proc = subprocess.Popen(
        [CLANGD_PATH, "--log=verbose", "-j=4"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Started clangd with verbose logging")
    print("Stderr will contain debug logs...")
    
    # Initialize
    send_request(proc, "initialize", {
        "processId": os.getpid(),
        "rootUri": f"file:///A/clangd-dev/hip-test",
        "capabilities": {}
    }, 1)
    
    resp = wait_for_response(proc, 1)
    print(f"Initialize response: {resp is not None}")
    
    # Open the file
    send_request(proc, "textDocument/didOpen", {
        "textDocument": {
            "uri": f"file://{TEST_FILE}",
            "languageId": "cpp",
            "version": 1,
            "text": file_content
        }
    }, None)
    
    print("File opened, waiting for indexing...")
    import time
    time.sleep(5)  # Wait for parsing
    
    # Try goto definition at line 9102 (load function call)
    # Line numbers are 0-based in LSP
    line = 9101  # 9102 in editor
    character = 8  # position of 'load'
    
    print(f"\nTrying goto definition at line {line+1}, char {character}")
    send_request(proc, "textDocument/definition", {
        "textDocument": {"uri": f"file://{TEST_FILE}"},
        "position": {"line": line, "character": character}
    }, 2)
    
    resp = wait_for_response(proc, 2, timeout=30)
    print(f"Goto definition response: {json.dumps(resp, indent=2) if resp else 'None'}")
    
    # Check stderr for debug logs
    print("\n--- Debug logs from clangd (last 50 lines) ---")
    proc.terminate()
    stderr_output = proc.stderr.read().decode()
    lines = stderr_output.strip().split('\n')
    for line in lines[-50:]:
        if 'locateASTReferent' in line or 'TemplateInstantiation' in line or 'Candidate' in line:
            print(line)
    
    proc.wait()

if __name__ == "__main__":
    main()

