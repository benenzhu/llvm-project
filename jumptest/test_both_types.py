#!/usr/bin/env python3
"""Test both float and double template instantiations."""

import json
import subprocess
import os
import time
import threading

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
PROJECT_ROOT = "/A/clangd-dev/jumptest"

class ClangdClient:
    def __init__(self):
        self.process = subprocess.Popen(
            [CLANGD_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_ROOT
        )
        self.request_id = 0
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()
        
    def _read_stderr(self):
        for line in self.process.stderr:
            pass  # Discard stderr
            
    def send_request(self, method, params):
        self.request_id += 1
        req_id = self.request_id
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        content = json.dumps(request)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.process.stdin.write(header.encode())
        self.process.stdin.write(content.encode())
        self.process.stdin.flush()
        return self._wait_for_response(req_id)
    
    def send_notification(self, method, params):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        content = json.dumps(msg)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.process.stdin.write(header.encode())
        self.process.stdin.write(content.encode())
        self.process.stdin.flush()
    
    def _wait_for_response(self, req_id, timeout=30):
        end_time = time.time() + timeout
        while time.time() < end_time:
            msg = self._receive_one(timeout=1)
            if msg is None:
                continue
            if "id" in msg and msg["id"] == req_id:
                return msg
        return None
    
    def _receive_one(self, timeout=10):
        headers = {}
        header_data = b""
        end_time = time.time() + timeout
        while time.time() < end_time:
            byte = self.process.stdout.read(1)
            if not byte:
                time.sleep(0.01)
                continue
            header_data += byte
            if header_data.endswith(b"\r\n\r\n"):
                break
        if not header_data.endswith(b"\r\n\r\n"):
            return None
        for line in header_data.decode().split("\r\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()
        content_length = int(headers.get("Content-Length", 0))
        if content_length > 0:
            content = self.process.stdout.read(content_length).decode()
            return json.loads(content)
        return None
    
    def initialize(self):
        resp = self.send_request("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file://{PROJECT_ROOT}",
            "capabilities": {}
        })
        self.send_notification("initialized", {})
        return resp
    
    def open_file(self, filepath):
        with open(filepath) as f:
            content = f.read()
        self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": f"file://{filepath}",
                "languageId": "cpp",
                "version": 1,
                "text": content
            }
        })
        time.sleep(2)
        return f"file://{filepath}"
    
    def goto_definition(self, uri, line, character):
        return self.send_request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character}
        })
    
    def hover(self, uri, line, character):
        return self.send_request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character}
        })
    
    def shutdown(self):
        try:
            self.send_request("shutdown", None)
            self.send_notification("exit", None)
        except:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=2)
        except:
            self.process.kill()


def test_both():
    print("=" * 60)
    print("Test: Both float and double instantiations")
    print("=" * 60)
    
    client = ClangdClient()
    
    try:
        client.initialize()
        uri = client.open_file(f"{PROJECT_ROOT}/jump2.cpp")
        
        # Test 1: Jump from float instantiation (line 18)
        print("\n=== TEST 1: From float (line 18) ===")
        client.goto_definition(uri, 17, 5)  # Jump from mma_AB(a)
        
        hover1 = client.hover(uri, 12, 17)  # Hover on 'now'
        if hover1 and "result" in hover1 and hover1["result"]:
            value = hover1["result"].get("contents", {}).get("value", "")
            if "Value = 4" in value and "8" not in value:
                print("   now hover: ✓ Value = 4")
            else:
                print(f"   now hover: ✗ {value}")
        
        goto1 = client.goto_definition(uri, 13, 12)  # Jump from mma_AB_base
        if goto1 and "result" in goto1:
            results = goto1["result"]
            if len(results) == 1 and results[0]["range"]["start"]["line"] == 1:
                print("   mma_AB_base jump: ✓ line 2 (float*)")
            else:
                print(f"   mma_AB_base jump: ✗ {len(results)} results")
        
        hover2 = client.hover(uri, 13, 12)
        if hover2 and "result" in hover2 and hover2["result"]:
            value = hover2["result"].get("contents", {}).get("value", "")
            if "float" in value:
                print("   mma_AB_base hover: ✓ float*")
            else:
                print(f"   mma_AB_base hover: ✗ {value[:100]}")
        
        # Test 2: Jump from double instantiation (line 20)
        print("\n=== TEST 2: From double (line 20) ===")
        client.goto_definition(uri, 19, 5)  # Jump from mma_AB(b)
        
        hover3 = client.hover(uri, 12, 17)  # Hover on 'now'
        if hover3 and "result" in hover3 and hover3["result"]:
            value = hover3["result"].get("contents", {}).get("value", "")
            if "Value = 8" in value and "4" not in value:
                print("   now hover: ✓ Value = 8")
            else:
                print(f"   now hover: ✗ {value}")
        
        goto2 = client.goto_definition(uri, 13, 12)  # Jump from mma_AB_base
        if goto2 and "result" in goto2:
            results = goto2["result"]
            if len(results) == 1 and results[0]["range"]["start"]["line"] == 2:
                print("   mma_AB_base jump: ✓ line 3 (double*)")
            else:
                print(f"   mma_AB_base jump: ✗ {len(results)} results, line {results[0]['range']['start']['line']+1 if results else '?'}")
        
        hover4 = client.hover(uri, 13, 12)
        if hover4 and "result" in hover4 and hover4["result"]:
            value = hover4["result"].get("contents", {}).get("value", "")
            if "double" in value:
                print("   mma_AB_base hover: ✓ double*")
            else:
                print(f"   mma_AB_base hover: ✗ {value[:100]}")
        
        print("\n" + "=" * 60)
        
    finally:
        client.shutdown()


if __name__ == "__main__":
    test_both()

