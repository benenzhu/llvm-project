#!/usr/bin/env python3
"""Test with debug logging."""

import json
import subprocess
import os
import time
import threading
import sys

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
            decoded = line.decode().strip()
            self.stderr_lines.append(decoded)
            # Print lines containing our debug markers
            if "locateASTReferent" in decoded or "CallFinder" in decoded:
                print(f"DEBUG: {decoded}", file=sys.stderr)
            
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
        # Print any remaining debug lines
        time.sleep(0.5)
        for line in self.stderr_lines:
            if "locateASTReferent" in line or "CallFinder" in line:
                print(f"DEBUG: {line}", file=sys.stderr)


def test():
    print("Test: Debug logging")
    
    client = ClangdClient()
    
    try:
        client.initialize()
        
        print("1. Opening jump2.cpp...")
        uri = client.open_file(f"{PROJECT_ROOT}/jump2.cpp")
        
        print("2. Jump from mma_AB(a) at line 18...")
        goto1 = client.goto_definition(uri, 17, 5)
        print(f"   Result: {goto1.get('result', [])}")
        
        print("3. Jump from mma_AB_base at line 14...")
        goto2 = client.goto_definition(uri, 13, 12)
        results = goto2.get('result', []) if goto2 else []
        print(f"   Found {len(results)} results")
        
        # Wait to collect debug output
        time.sleep(1)
        
    finally:
        print("\n=== Shutting down ===")
        client.shutdown()


if __name__ == "__main__":
    test()

