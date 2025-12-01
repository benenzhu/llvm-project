#!/usr/bin/env python3
"""Test jump from line 7814 kittens::load."""
import subprocess
import json
import time
import sys

CLANGD = "/A/clangd-dev/build/bin/clangd"
FILE = "/A/clangd-dev/hip-test/read.cpp"

proc = subprocess.Popen(
    [CLANGD, "-j=4"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)

def send(method, params, req_id=None):
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    if req_id:
        msg["id"] = req_id
    body = json.dumps(msg)
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n{body}".encode())
    proc.stdin.flush()

def recv():
    header = b""
    while not header.endswith(b"\r\n\r\n"):
        header += proc.stdout.read(1)
    cl = int([l for l in header.decode().split('\r\n') if l.startswith('Content-Length')][0].split(':')[1])
    return json.loads(proc.stdout.read(cl).decode())

def wait_for(req_id, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        msg = recv()
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

print("Waiting for indexing...", file=sys.stderr)
time.sleep(15)

# Test: Jump from line 7814 kittens::load
# Line 7814: "    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);"
# 'load' starts at column 14
print("\n=== Testing: Jump from kittens::load at line 7814 ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 7813, "character": 14}
}, 2)

resp = wait_for(2, timeout=30)
if resp and resp.get('result'):
    result = resp['result']
    if isinstance(result, list):
        print(f"Results: {len(result)}")
        if len(result) == 1:
            print("SUCCESS: Only 1 result!")
        else:
            print(f"FAIL: Expected 1, got {len(result)}")
        for i, loc in enumerate(result[:5]):
            line = loc.get('range', {}).get('start', {}).get('line', -1) + 1
            print(f"  [{i}] line {line}")
else:
    print("TIMEOUT or no result")

proc.terminate()

