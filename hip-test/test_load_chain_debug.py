#!/usr/bin/env python3
"""
Test chained jump in read.cpp:
  Step 1: Jump from G::load (line 9064) -> line 7813
  Step 2: Jump from kittens::load (line 7814) -> should return ONLY ONE result

This tests that template context is properly preserved and used to filter
overloaded function results when jumping within template functions.
"""
import subprocess
import json
import time
import sys

CLANGD = "/A/clangd-dev/build/bin/clangd"
FILE = "/A/clangd-dev/hip-test/read.cpp"

# Redirect stderr to file to avoid blocking
log_file = open("/tmp/clangd_log.txt", "w")
proc = subprocess.Popen(
    [CLANGD, "-j=4", "--log=verbose"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_file,
)

def send(method, params, req_id=None):
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    if req_id: msg["id"] = req_id
    body = json.dumps(msg)
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n{body}".encode())
    proc.stdin.flush()

def recv():
    header = b""
    while not header.endswith(b"\r\n\r\n"):
        c = proc.stdout.read(1)
        if not c: return None
        header += c
    cl = int([l for l in header.decode().split('\r\n') if 'Content-Length' in l][0].split(':')[1])
    return json.loads(proc.stdout.read(cl).decode())

def wait_response(req_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        msg = recv()
        if msg and msg.get('id') == req_id:
            return msg
    return None

# Initialize
with open(FILE) as f:
    content = f.read()

send("initialize", {"processId": 1, "rootUri": f"file://{FILE}", "capabilities": {}}, 1)
wait_response(1)
send("initialized", {})
send("textDocument/didOpen", {"textDocument": {"uri": f"file://{FILE}", "languageId": "cpp", "version": 1, "text": content}})

print("Waiting for indexing...", file=sys.stderr)
time.sleep(12)

# Step 1: Jump from G::load (line 9064, col 7 = "load")
print("=== Step 1: Jump from G::load (line 9064) ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 9063, "character": 7}  # 0-indexed: line 9064, col 8
}, 2)
resp1 = wait_response(2)
if resp1 and "result" in resp1:
    results = resp1["result"]
    print(f"Results: {len(results)}")
    for i, r in enumerate(results[:5]):
        line = r.get("targetRange", r.get("range", {})).get("start", {}).get("line", -1)
        print(f"  [{i}] line {line + 1}")
else:
    print("FAIL: No response for Step 1")
    proc.terminate()
    sys.exit(1)

time.sleep(1)

# Step 2: Jump from kittens::load (line 7814, col 14 = "load")
print("\n=== Step 2: Jump from kittens::load (line 7814) ===")
send("textDocument/definition", {
    "textDocument": {"uri": f"file://{FILE}"},
    "position": {"line": 7813, "character": 14}  # 0-indexed: line 7814, col 15
}, 3)
resp2 = wait_response(3)
if resp2 and "result" in resp2:
    results = resp2["result"]
    print(f"Results: {len(results)}")
    if len(results) == 1:
        print("SUCCESS!")
    else:
        print(f"FAIL: Expected 1 result, got {len(results)}")
    for i, r in enumerate(results[:5]):
        line = r.get("targetRange", r.get("range", {})).get("start", {}).get("line", -1)
        print(f"  [{i}] line {line + 1}")
else:
    print("FAIL: No response for Step 2")

proc.terminate()
