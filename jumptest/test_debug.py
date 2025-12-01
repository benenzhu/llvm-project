#!/usr/bin/env python3
"""Debug test for template context."""

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


def test_debug():
    print("=" * 60)
    print("Debug: Template context flow")
    print("=" * 60)
    
    client = ClangdClient()
    
    try:
        client.initialize()
        
        print("\n1. Opening jump2.cpp...")
        uri = client.open_file(f"{PROJECT_ROOT}/jump2.cpp")
        
        # First jump
        print("\n2. First jump: mma_AB(a) at line 18...")
        goto1 = client.goto_definition(uri, 17, 4)
        print(f"   Jump result: {goto1}")
        
        # Wait a bit
        time.sleep(0.5)
        
        # Hover immediately after jump
        print("\n3. Hover on 'now' at line 13 (immediately after jump)...")
        hover1 = client.hover(uri, 12, 16)
        if hover1 and "result" in hover1 and hover1["result"]:
            contents = hover1["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            # Look for Value line
            for line in value.split('\n'):
                if 'Value' in line:
                    print(f"   Found: {line.strip()}")
                    break
            else:
                print("   No Value found in hover")
        else:
            print(f"   No hover result")
        
        # Second jump (from same call site)
        print("\n4. Second jump: mma_AB(a) at line 18 again...")
        goto2 = client.goto_definition(uri, 17, 4)
        
        time.sleep(0.5)
        
        # Hover again
        print("\n5. Hover on 'now' again...")
        hover2 = client.hover(uri, 12, 16)
        if hover2 and "result" in hover2 and hover2["result"]:
            contents = hover2["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            for line in value.split('\n'):
                if 'Value' in line:
                    print(f"   Found: {line.strip()}")
                    break
            else:
                print("   No Value found in hover")
        
        # Jump to mma_AB_base
        print("\n6. Jump to mma_AB_base at line 14...")
        goto3 = client.goto_definition(uri, 13, 4)
        if goto3 and "result" in goto3 and goto3["result"]:
            results = goto3["result"] if isinstance(goto3["result"], list) else [goto3["result"]]
            print(f"   Found {len(results)} definition(s)")
            for r in results:
                line = r.get("range", {}).get("start", {}).get("line", 0) + 1
                print(f"   → Line {line}")
        
    finally:
        print("\n=== Shutting down ===")
        client.shutdown()


if __name__ == "__main__":
    test_debug()

