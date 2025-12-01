#!/usr/bin/env python3
"""Debug test with logs."""
import subprocess
import json
import time
import sys
import os

CLANGD = "/A/clangd-dev/build/bin/clangd"
FILE = "/A/clangd-dev/hip-test/read.cpp"

# Start with verbose logging
proc = subprocess.Popen(
    [CLANGD, "-j=4", "--log=verbose"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

def send(method, params, req_id=None):
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    if req_id:
        msg["id"] = req_id
    body = json.dumps(msg)
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n{body}".encode())
    proc.stdin.flush()

def recv_msg():
    header = b""
    while not header.endswith(b"\r\n\r\n"):
        c = proc.stdout.read(1)
        if not c:
            return None
        header += c
    cl = int([l for l in header.decode().split('\r\n') if l.startswith('Content-Length')][0].split(':')[1])
    return json.loads(proc.stdout.read(cl).decode())

def wait_for(req_id, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        msg = recv_msg()
        if not msg:
            continue
        if msg.get('id') == req_id:
            return msg
    return None

# Initialize
send("initialize", {"processId": 1, "rootUri": "file:///A/clangd-dev/hip-test", "capabilities": {}}, 1)
wait_for(1)
send("initialized", {})

# Open file
with open(FILE) as f:
    content = f.read()
send("textDocument/didOpen", {
    "textDocument": {"uri": f"file://{FILE}", "languageId": "cpp", "version": 1, "text": content}
})

print("Waiting 10s for indexing...", file=sys.stderr)
time.sleep(10)

# Step 1: Jump from line 9064 G::load
print("=== Step 1: Jump from G::load at line 9064 ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 9063, "character": 7}
}, 2)
resp = wait_for(2, timeout=20)
if resp and resp.get('result'):
    result = resp['result']
    print(f"Results: {len(result)}")
    for i, loc in enumerate(result[:3]):
        line = loc.get('range', {}).get('start', {}).get('line', -1) + 1
        print(f"  [{i}] line {line}")

time.sleep(1)

# Step 2: Jump from line 7814 kittens::load
print("\n=== Step 2: Jump from kittens::load at line 7814 ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 7813, "character": 14}
}, 3)
resp = wait_for(3, timeout=20)
if resp and resp.get('result'):
    result = resp['result']
    print(f"Results: {len(result)}")
    for i, loc in enumerate(result[:5]):
        line = loc.get('range', {}).get('start', {}).get('line', -1) + 1
        print(f"  [{i}] line {line}")

proc.terminate()

# Read stderr for logs
stderr = proc.stderr.read().decode()
print("\n=== Relevant logs ===", file=sys.stderr)
for line in stderr.split('\n'):
    if any(k in line for k in ["Candidates", "Instantiation", "locateAST", "Template", "filtering", "Found"]):
        print(line, file=sys.stderr)
