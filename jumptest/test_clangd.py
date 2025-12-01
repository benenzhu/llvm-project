#!/usr/bin/env python3
"""
Test script for clangd template context functionality.
"""

import json
import subprocess
import sys
import os
import time
import threading

CLANGD_PATH = "/A/clangd-dev/build/bin/clangd"
PROJECT_ROOT = "/A/clangd-dev/jumptest"

class ClangdClient:
    def __init__(self):
        self.process = subprocess.Popen(
            [CLANGD_PATH, "--log=verbose", f"--compile-commands-dir={PROJECT_ROOT}"],
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
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }
        self._send(request)
        return self._wait_for_response(req_id)
    
    def send_notification(self, method, params):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self._send(notification)
        
    def _send(self, message):
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.process.stdin.write(header.encode())
        self.process.stdin.write(content.encode())
        self.process.stdin.flush()
    
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
    
    def _wait_for_response(self, req_id, timeout=30):
        end_time = time.time() + timeout
        while time.time() < end_time:
            msg = self._receive_one(timeout=1)
            if msg is None:
                continue
            if "id" in msg and msg["id"] == req_id:
                return msg
        return None
    
    def initialize(self):
        response = self.send_request("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file://{PROJECT_ROOT}",
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {}
                }
            }
        })
        self.send_notification("initialized", {})
        return response
    
    def open_file(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        uri = f"file://{filepath}"
        self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "cpp",
                "version": 1,
                "text": content
            }
        })
        time.sleep(2)
        return uri
    
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
    
    def print_stderr(self, filter_str=None):
        print("\n--- Clangd stderr (relevant) ---")
        for line in self.stderr_lines[-30:]:
            if filter_str is None or filter_str in line.lower():
                print(f"  {line}")
        print("---")


def test_context_flow():
    """Test the full flow: jump -> hover -> jump"""
    print("=" * 60)
    print("Test: Full context flow (jump.cpp -> jump2.h -> jump1.h)")
    print("=" * 60)
    
    client = ClangdClient()
    
    try:
        client.initialize()
        
        print("\n1. Opening jump.cpp...")
        jump_cpp_uri = client.open_file(f"{PROJECT_ROOT}/jump.cpp")
        
        print("\n2. Jump from mma_AB(a) at line 5, col 4...")
        goto1 = client.goto_definition(jump_cpp_uri, 4, 4)
        
        if goto1 and "result" in goto1 and goto1["result"]:
            results = goto1["result"] if isinstance(goto1["result"], list) else [goto1["result"]]
            for r in results:
                uri = r.get("uri", r.get("targetUri", ""))
                rng = r.get("range", r.get("targetRange", {}))
                print(f"   → Jumped to: {uri}")
                print(f"     Line {rng.get('start', {}).get('line', 0) + 1}")
        
        # Now test hover on 'now' in jump2.h (without opening it separately)
        jump2_uri = f"file://{PROJECT_ROOT}/jump2.h"
        
        print(f"\n3. Hover on 'now' at jump2.h line 3, col 16...")
        hover1 = client.hover(jump2_uri, 2, 16)
        
        if hover1 and "result" in hover1 and hover1["result"]:
            contents = hover1["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            print(f"   Hover result:")
            for line in value.split('\n')[:10]:
                print(f"      {line}")
            
            if "Value" in value:
                print("   ✓ Value shown!")
            else:
                print("   ✗ No Value in result")
        else:
            print(f"   ✗ No hover result: {hover1}")
        
        print(f"\n4. Jump from mma_AB_base at jump2.h line 4, col 4...")
        goto2 = client.goto_definition(jump2_uri, 3, 4)
        
        if goto2 and "result" in goto2 and goto2["result"]:
            results = goto2["result"] if isinstance(goto2["result"], list) else [goto2["result"]]
            print(f"   Found {len(results)} definition(s):")
            for r in results:
                uri = r.get("uri", r.get("targetUri", ""))
                rng = r.get("range", r.get("targetRange", {}))
                line_num = rng.get('start', {}).get('line', 0) + 1
                print(f"   → {uri} line {line_num}")
                
                # Check if it's the right overload
                if "jump1.h" in uri:
                    if line_num == 2:
                        print("   ✓ Correct! mma_AB_base(float*)")
                    elif line_num == 1:
                        print("   ✗ Wrong: mma_AB_base(int*)")
                    elif line_num == 3:
                        print("   ✗ Wrong: mma_AB_base(double*)")
        else:
            print(f"   ✗ No definition found: {goto2}")
        
        client.print_stderr("astworker")
        
    finally:
        client.shutdown()


def test_without_prior_jump():
    """Test hover/jump on header without first jumping from cpp"""
    print("\n" + "=" * 60)
    print("Test: Direct header access (no prior jump from cpp)")
    print("=" * 60)
    
    client = ClangdClient()
    
    try:
        client.initialize()
        
        print("\n1. Opening jump.cpp (for compilation context)...")
        client.open_file(f"{PROJECT_ROOT}/jump.cpp")
        
        print("\n2. Opening jump2.h directly...")
        jump2_uri = client.open_file(f"{PROJECT_ROOT}/jump2.h")
        
        print(f"\n3. Hover on 'now' at line 3...")
        hover = client.hover(jump2_uri, 2, 16)
        
        if hover and "result" in hover and hover["result"]:
            contents = hover["result"].get("contents", {})
            value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
            print(f"   Result:")
            for line in value.split('\n')[:8]:
                print(f"      {line}")
        else:
            print(f"   ✗ No result")
        
        print(f"\n4. Jump from mma_AB_base at line 4...")
        goto = client.goto_definition(jump2_uri, 3, 4)
        
        if goto and "result" in goto and goto["result"]:
            results = goto["result"] if isinstance(goto["result"], list) else [goto["result"]]
            print(f"   Found {len(results)} definition(s)")
            for r in results:
                uri = r.get("uri", "")
                rng = r.get("range", {})
                print(f"   → {uri} line {rng.get('start', {}).get('line', 0) + 1}")
        else:
            print(f"   ✗ No definition (expected - no context)")
        
        client.print_stderr("astworker")
        
    finally:
        client.shutdown()


if __name__ == "__main__":
    test_context_flow()
    test_without_prior_jump()
