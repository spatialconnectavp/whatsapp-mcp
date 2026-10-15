"""Subprocess regressions for client-triggered MCP shutdown."""

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signal and pipe semantics")


def start_server(tmp_path, *args):
    env = os.environ.copy()
    env.update(
        WHATSAPP_MCP_TRANSPORT="stdio",
        WHATSAPP_DB_PATH=str(tmp_path / "messages.db"),
        WHATSMEOW_DB_PATH=str(tmp_path / "session.db"),
    )
    return subprocess.Popen(
        [sys.executable, *args],
        cwd=SERVER_DIR,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def read_reply(process):
    assert select.select([process.stdout], [], [], 10)[0], "Server did not respond"
    return process.stdout.readline()


def cleanup(process):
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_exits_without_abort_when_stdin_thread_holds_lock(tmp_path, signum):
    # A background pipe reader can still own stdin's buffered lock at shutdown.
    # The old sys.exit handler runs finalizers and aborts in _enter_buffered_busy.
    process = start_server(
        tmp_path,
        "-c",
        """
import signal
import sys
import threading
import time
from main import shutdown_handler
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)
threading.Thread(target=lambda: sys.stdin.buffer.read(1), daemon=True).start()
time.sleep(0.1)
print('ready', flush=True)
while True:
    time.sleep(1)
""",
    )
    try:
        assert read_reply(process) == b"ready\n"
        process.send_signal(signum)
        assert process.wait(timeout=5) == 0, process.stderr.read().decode(errors="replace")
    finally:
        cleanup(process)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_live_stdio_server_stops_promptly_with_input_pipe_open(tmp_path, signum):
    process = start_server(tmp_path, "main.py")
    try:
        messages = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "shutdown-test", "version": "1"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        for message in messages:
            process.stdin.write((json.dumps(message) + "\n").encode())
            if "id" in message:
                reply = json.loads(read_reply(process))
                assert reply["id"] == message["id"]
                assert "result" in reply
        # Let the transport start its next blocking read before signalling.
        time.sleep(0.1)
        process.send_signal(signum)
        assert process.wait(timeout=5) == 0, process.stderr.read().decode(errors="replace")
    finally:
        cleanup(process)


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
def test_network_shutdown_keeps_normal_python_cleanup(tmp_path, transport):
    process = start_server(
        tmp_path,
        "-c",
        f"""
import atexit
import signal
import time
from functools import partial
from main import shutdown_handler
atexit.register(lambda: print('cleanup', flush=True))
signal.signal(signal.SIGTERM, partial(shutdown_handler, transport={transport!r}))
print('ready', flush=True)
while True:
    time.sleep(1)
""",
    )
    try:
        assert read_reply(process) == b"ready\n"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0, process.stderr.read().decode(errors="replace")
        assert process.stdout.read() == b"cleanup\n"
    finally:
        cleanup(process)


def test_stdio_eof_returns_queued_reply_before_exit(tmp_path):
    process = start_server(tmp_path, "main.py")
    try:
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "eof-test", "version": "1"},
            },
        }
        process.stdin.write((json.dumps(initialize) + "\n").encode())
        assert "result" in json.loads(read_reply(process))
        for message in [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]:
            process.stdin.write((json.dumps(message) + "\n").encode())
        process.stdin.close()
        reply = json.loads(read_reply(process))
        assert reply["id"] == 2
        assert reply["result"]["tools"]
        assert process.wait(timeout=5) == 0, process.stderr.read().decode(errors="replace")
    finally:
        cleanup(process)
