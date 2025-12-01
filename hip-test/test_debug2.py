#!/usr/bin/env python3
import subprocess, json, time, sys

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

def wait_for(req_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        msg = recv()
        if msg and msg.get('id') == req_id: return msg
    return None

send("initialize", {"processId": 1, "rootUri": "file:///A/clangd-dev/hip-test", "capabilities": {}}, 1)
wait_for(1)
send("initialized", {})

with open(FILE) as f: content = f.read()
send("textDocument/didOpen", {"textDocument": {"uri": f"file://{FILE}", "languageId": "cpp", "version": 1, "text": content}})
time.sleep(10)

print("=== Step 1: Jump from G::load (line 9064) ===")
send("textDocument/definition", {"textDocument": {"uri": f"file://{FILE}"}, "position": {"line": 9063, "character": 7}}, 2)
resp = wait_for(2, timeout=20)
if resp and resp.get('result'):
    print(f"Results: {len(resp['result'])}")
    for i, loc in enumerate(resp['result'][:3]):
        print(f"  [{i}] line {loc.get('range', {}).get('start', {}).get('line', -1) + 1}")

time.sleep(1)

print("\n=== Step 2: Jump from kittens::load (line 7814) ===")
send("textDocument/definition", {"textDocument": {"uri": f"file://{FILE}"}, "position": {"line": 7813, "character": 14}}, 3)
resp = wait_for(3, timeout=20)
if resp and resp.get('result'):
    result = resp['result']
    print(f"Results: {len(result)}")
    if len(result) == 1: print("SUCCESS!")
    else: print(f"FAIL: Expected 1")
    for i, loc in enumerate(result[:5]):
        print(f"  [{i}] line {loc.get('range', {}).get('start', {}).get('line', -1) + 1}")

proc.terminate()
log_file.close()

