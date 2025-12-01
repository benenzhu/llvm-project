#!/usr/bin/env python3
"""Test single file template jump."""

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


def test_single_file():
    print("=" * 60)
    print("Test: Single file template jump (jump2.cpp)")
    print("=" * 60)
    
    client = ClangdClient()
    
    try:
        client.initialize()
        
        print("\n1. Opening jump2.cpp...")
        uri = client.open_file(f"{PROJECT_ROOT}/jump2.cpp")
        
        # Test 1: Jump from line 18 (mma_AB(a) where a is rt<float>)
        print("\n2. Jump from mma_AB(a) at line 18 (T should be float)...")
        goto1 = client.goto_definition(uri, 17, 4)  # 0-indexed: line 17
        
        if goto1 and "result" in goto1 and goto1["result"]:
            results = goto1["result"]
            for r in results:
                target_line = r.get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → Jumped to line {target_line}")
        
        # Now hover on 'now' at line 13
        print("\n3. Hover on 'now' at line 13 (should show Value = 4)...")
        hover1 = client.hover(uri, 12, 16)  # 0-indexed: line 12
        
        if hover1 and "result" in hover1 and hover1["result"]:
            contents = hover1["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            print("   Hover result:")
            for line in value.split('\n')[:10]:
                print(f"      {line}")
            
            if "4, 8" in value:
                print("   ✗ Shows both values - context not working")
            elif "Value = 4" in value or "Value: 4" in value:
                print("   ✓ Shows correct value for float!")
            elif "Value" not in value:
                print("   ✗ No Value shown")
        else:
            print(f"   ✗ No result")
        
        # Jump from mma_AB_base at line 14
        print("\n4. Jump from mma_AB_base at line 14 (should go to float* overload)...")
        goto2 = client.goto_definition(uri, 13, 4)  # 0-indexed: line 13
        
        if goto2 and "result" in goto2 and goto2["result"]:
            results = goto2["result"] if isinstance(goto2["result"], list) else [goto2["result"]]
            print(f"   Found {len(results)} definition(s):")
            for r in results:
                line = r.get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → Line {line}")
                if line == 2:
                    print("      ✓ Correct: mma_AB_base(float*)")
                elif line == 1:
                    print("      ✗ Wrong: mma_AB_base(int*)")
                elif line == 3:
                    print("      ✗ Wrong: mma_AB_base(double*)")
            if len(results) == 3:
                print("   ✗ Shows all 3 overloads - context not working")
        else:
            print(f"   ✗ No result")
        
        # Test 2: Jump from line 20 (mma_AB(b) where b is rt<double>)
        print("\n" + "=" * 60)
        print("\n5. Jump from mma_AB(b) at line 20 (T should be double)...")
        goto3 = client.goto_definition(uri, 19, 4)  # 0-indexed: line 19
        
        if goto3 and "result" in goto3 and goto3["result"]:
            results = goto3["result"]
            for r in results:
                target_line = r.get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → Jumped to line {target_line}")
        
        # Hover on 'now' again - should show 8 for double
        print("\n6. Hover on 'now' at line 13 (should show Value = 8)...")
        hover2 = client.hover(uri, 12, 16)
        
        if hover2 and "result" in hover2 and hover2["result"]:
            contents = hover2["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            print("   Hover result:")
            for line in value.split('\n')[:10]:
                print(f"      {line}")
            
            if "4, 8" in value:
                print("   ✗ Shows both values - context not working")
            elif "Value = 8" in value or "Value: 8" in value:
                print("   ✓ Shows correct value for double!")
            elif "Value" not in value:
                print("   ✗ No Value shown")
        
    finally:
        print("\n=== Shutting down ===")
        client.shutdown()


if __name__ == "__main__":
    test_single_file()

