#!/usr/bin/env python3
"""Test chained jump: 9064 -> 7814 -> ?
This tests that template context is preserved through the chain.
"""
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
        c = proc.stdout.read(1)
        if not c:
            return None
        header += c
    cl = int([l for l in header.decode().split('\r\n') if l.startswith('Content-Length')][0].split(':')[1])
    return json.loads(proc.stdout.read(cl).decode())

def wait_for(req_id, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        msg = recv()
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

print("Waiting for indexing...", file=sys.stderr)
time.sleep(12)

# Step 1: Jump from line 9064 G::load to line 7814
# Line 9064: "    G::load(As, g.a, {0, 0, row, 0});"
# 'load' is at column 7
print("\n=== Step 1: Jump from G::load at line 9064 ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 9063, "character": 7}  # 0-based
}, 2)

resp = wait_for(2, timeout=30)
step1_target = None
if resp and resp.get('result'):
    result = resp['result']
    if isinstance(result, list) and len(result) > 0:
        print(f"Results: {len(result)}")
        for i, loc in enumerate(result[:3]):
            line = loc.get('range', {}).get('start', {}).get('line', -1) + 1
            print(f"  [{i}] line {line}")
            if i == 0:
                step1_target = line
else:
    print("No result")

time.sleep(2)

# Step 2: Jump from line 7814 kittens::load
# Line 7814: "    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);"
# 'load' is at column 14
print("\n=== Step 2: Jump from kittens::load at line 7814 ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 7813, "character": 14}  # 0-based
}, 3)

resp = wait_for(3, timeout=30)
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
    print("No result")

proc.terminate()

