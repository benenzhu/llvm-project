#!/usr/bin/env python3
"""Test simulating actual user flow from the log."""

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
            [CLANGD_PATH, "--log=verbose"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_ROOT
        )
        self.request_id = 0
        self.stderr_lines = []
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()
        
    def _read_stderr(self):
        for line in self.process.stderr:
            self.stderr_lines.append(line.decode().strip())
            
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


def test_user_flow():
    """Simulate the exact user flow from the log."""
    print("=" * 60)
    print("Test: Simulating user flow from log")
    print("=" * 60)
    
    client = ClangdClient()
    
    try:
        client.initialize()
        
        print("\n1. Opening jump2.cpp...")
        uri = client.open_file(f"{PROJECT_ROOT}/jump2.cpp")
        
        # Step 1: Jump from mma_AB(a) at line 18 (0-indexed: 17)
        print("\n2. Jump from mma_AB(a) at line 18, char 5...")
        goto1 = client.goto_definition(uri, 17, 5)
        if goto1 and "result" in goto1:
            target = goto1["result"][0] if goto1["result"] else None
            if target:
                line = target.get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → Jumped to line {line}")
        
        # Step 2: User clicks on 'now' at line 13 (like clicking to see definition)
        # This is what happened in the log - user clicked definition on 'now'
        print("\n3. Jump on 'now' at line 13, char 17 (user clicks to see definition)...")
        goto2 = client.goto_definition(uri, 12, 17)
        if goto2 and "result" in goto2:
            target = goto2["result"][0] if goto2["result"] else None
            if target:
                line = target.get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → now definition at line {line}")
        
        # Step 3: Hover on 'now' - this is where the problem was
        print("\n4. Hover on 'now' at line 13, char 17 (should show Value = 4)...")
        hover1 = client.hover(uri, 12, 17)
        if hover1 and "result" in hover1 and hover1["result"]:
            contents = hover1["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            # Find Value line
            for line in value.split('\n'):
                if 'Value' in line:
                    print(f"   Found: {line.strip()}")
                    if "4, 8" in line:
                        print("   ✗ FAIL: Shows both values - context was lost")
                    elif "4" in line and "8" not in line:
                        print("   ✓ PASS: Shows only float value!")
                    break
        else:
            print("   ✗ No hover result")
        
        # Step 4: Jump to mma_AB_base
        print("\n5. Jump from mma_AB_base at line 14, char 12...")
        goto3 = client.goto_definition(uri, 13, 12)
        if goto3 and "result" in goto3:
            results = goto3["result"] if isinstance(goto3["result"], list) else [goto3["result"]]
            print(f"   Found {len(results)} definition(s)")
            if len(results) == 1:
                line = results[0].get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → Line {line}")
                if line == 2:
                    print("   ✓ PASS: Correct overload (float*)")
                else:
                    print(f"   ✗ FAIL: Wrong overload")
            elif len(results) == 3:
                print("   ✗ FAIL: Shows all 3 overloads")
        
        # Also test hover on mma_AB_base
        print("\n6. Hover on mma_AB_base (should show float* parameter)...")
        hover2 = client.hover(uri, 13, 12)
        if hover2 and "result" in hover2 and hover2["result"]:
            contents = hover2["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            if "float" in value:
                print("   ✓ PASS: Shows float* parameter")
            elif "double" in value:
                print("   ✗ FAIL: Shows double* parameter")
            elif "int" in value:
                print("   ✗ FAIL: Shows int* parameter")
            # Print the parameter line
            for line in value.split('\n'):
                if '*' in line and 'a' in line:
                    print(f"   Parameter: {line.strip()}")
                    break
        
    finally:
        print("\n=== Shutting down ===")
        client.shutdown()


if __name__ == "__main__":
    test_user_flow()

