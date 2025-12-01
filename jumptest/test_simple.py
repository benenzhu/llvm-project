#!/usr/bin/env python3
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
            print(f"[STDERR] {line.decode().strip()}")
            
    def send_request(self, method, params):
        self.request_id += 1
        req_id = self.request_id
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        content = json.dumps(request)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        print(f"[SEND] {method} id={req_id}")
        self.process.stdin.write(header.encode())
        self.process.stdin.write(content.encode())
        self.process.stdin.flush()
        return self._wait_for_response(req_id)
    
    def send_notification(self, method, params):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        content = json.dumps(msg)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        print(f"[SEND] {method} (notification)")
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
                print(f"[RECV] Response for {req_id}: {json.dumps(msg)[:200]}")
                return msg
        print(f"[TIMEOUT] No response for {req_id}")
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


def main():
    client = ClangdClient()
    
    try:
        # Initialize
        client.send_request("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file://{PROJECT_ROOT}",
            "capabilities": {}
        })
        client.send_notification("initialized", {})
        
        # Open jump.cpp
        with open(f"{PROJECT_ROOT}/jump.cpp") as f:
            cpp_content = f.read()
        
        client.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": f"file://{PROJECT_ROOT}/jump.cpp",
                "languageId": "cpp",
                "version": 1,
                "text": cpp_content
            }
        })
        
        time.sleep(3)  # Wait for indexing
        
        # Jump to template definition
        print("\n=== Jump from mma_AB(a) ===")
        goto1 = client.send_request("textDocument/definition", {
            "textDocument": {"uri": f"file://{PROJECT_ROOT}/jump.cpp"},
            "position": {"line": 4, "character": 4}
        })
        
        if goto1 and "result" in goto1:
            print(f"Jump result: {goto1['result']}")
        
        # Hover on now in jump2.h (without opening it)
        print("\n=== Hover on 'now' in jump2.h ===")
        hover1 = client.send_request("textDocument/hover", {
            "textDocument": {"uri": f"file://{PROJECT_ROOT}/jump2.h"},
            "position": {"line": 2, "character": 16}
        })
        
        if hover1:
            print(f"Hover result: {json.dumps(hover1, indent=2)[:500]}")
        
    finally:
        print("\n=== Shutting down ===")
        client.shutdown()


if __name__ == "__main__":
    main()

