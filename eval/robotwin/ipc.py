from __future__ import annotations

import pickle
import os
import socket
import struct
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from eval.shared_policy_server import PROTOCOL_VERSION


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 512 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(min(remaining, 4 * 1024 * 1024))
        if not chunk:
            raise ConnectionError("IPC peer closed before the message completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"IPC message is too large: {len(encoded)} bytes")
    sock.sendall(_HEADER.pack(len(encoded)))
    sock.sendall(encoded)


def recv_message(sock: socket.socket) -> Any:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"IPC peer declared an oversized message: {size} bytes")
    return pickle.loads(_recv_exact(sock, size))


class UnixModelClient:
    def __init__(
        self,
        socket_path: str,
        timeout_sec: float = 1800.0,
        profiler: Any = None,
        client_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.socket_path = str(Path(socket_path))
        self.profiler = profiler
        self.client_id = str(
            client_id or f"robotwin-client-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.metadata = dict(metadata or {})
        self._request_sequence = 0
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(float(timeout_sec))
        with self.profile_section("setup/ipc_connect"):
            self.sock.connect(self.socket_path)
        with self.profile_section("setup/ipc_open_session"):
            send_message(
                self.sock,
                {
                    "type": "hello",
                    "protocol_version": PROTOCOL_VERSION,
                    "client_id": self.client_id,
                    "metadata": {
                        "benchmark": "robotwin",
                        **self.metadata,
                    },
                },
            )
            hello = recv_message(self.sock)
        if not isinstance(hello, dict) or not hello.get("ok", False):
            raise RuntimeError(
                f"Model server rejected client {self.client_id!r}: "
                f"{hello.get('error') if isinstance(hello, dict) else hello!r}"
            )
        if hello.get("type") != "hello_ack":
            raise RuntimeError(f"Malformed model server hello response: {hello!r}")
        if int(hello.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise RuntimeError(f"Model server protocol mismatch: {hello!r}")
        self.server_id = str(hello["server_id"])
        self.session_id = str(hello["session_id"])
        self.sim_seconds = 0.0
        self._client_timing: dict[str, float | int] = {}
        self._reset_client_timing()

    def profile_section(self, name: str):
        return self.profiler.section(name) if self.profiler is not None else nullcontext()

    def _reset_client_timing(self) -> None:
        self._client_timing = {
            "get_action_rpc_s": 0.0,
            "get_action_server_s": 0.0,
            "get_action_transport_s": 0.0,
            "get_action_queue_wait_s": 0.0,
            "get_action_handler_s": 0.0,
            "get_action_calls": 0,
            "update_obs_rpc_s": 0.0,
            "update_obs_server_s": 0.0,
            "update_obs_transport_s": 0.0,
            "update_obs_queue_wait_s": 0.0,
            "update_obs_handler_s": 0.0,
            "update_obs_calls": 0,
            "ipc_send_s": 0.0,
            "ipc_wait_s": 0.0,
            "policy_queue_wait_s": 0.0,
            "policy_handler_s": 0.0,
            "get_obs_s": 0.0,
            "get_obs_calls": 0,
            "observation_filter_s": 0.0,
            "observation_filter_calls": 0,
            "take_action_s": 0.0,
            "action_steps": 0,
        }

    def add_client_timing(
        self,
        name: str,
        elapsed: float,
        *,
        count_name: str | None = None,
        count: int = 1,
    ) -> None:
        self._client_timing[name] = float(self._client_timing.get(name, 0.0)) + float(elapsed)
        if count_name is not None:
            self._client_timing[count_name] = int(self._client_timing.get(count_name, 0)) + int(count)

    def call(
        self,
        command: str,
        payload: Any = None,
        *,
        track_timing: bool = True,
    ) -> Any:
        command = str(command)
        self._request_sequence += 1
        request_id = f"{self.client_id}:{self._request_sequence}"
        call_started = time.perf_counter()
        send_started = time.perf_counter()
        with self.profile_section(f"ipc/{command}/send"):
            send_message(
                self.sock,
                {
                    "type": "request",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": request_id,
                    "command": command,
                    "payload": payload,
                },
            )
        send_elapsed = time.perf_counter() - send_started
        wait_started = time.perf_counter()
        with self.profile_section(f"ipc/{command}/wait_response"):
            response = recv_message(self.sock)
        wait_elapsed = time.perf_counter() - wait_started
        call_elapsed = time.perf_counter() - call_started
        if not isinstance(response, dict):
            raise RuntimeError(f"Malformed model response: {type(response)!r}")
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"Model response request_id mismatch: expected {request_id!r}, "
                f"got {response.get('request_id')!r}"
            )
        if not response.get("ok", False):
            raise RuntimeError(
                f"Model server command {command!r} failed: {response.get('error')}\n"
                f"fatal={response.get('fatal', False)}\n{response.get('traceback', '')}"
            )
        if track_timing:
            queue_wait_s = float(response.get("queue_wait_s", 0.0) or 0.0)
            handler_s = float(
                response.get("handler_s", response.get("server_call_s", 0.0)) or 0.0
            )
            self.add_client_timing(f"{command}_rpc_s", call_elapsed)
            self.add_client_timing(f"{command}_server_s", handler_s)
            self.add_client_timing(f"{command}_queue_wait_s", queue_wait_s)
            self.add_client_timing(f"{command}_handler_s", handler_s)
            self.add_client_timing(
                f"{command}_transport_s",
                max(call_elapsed - queue_wait_s - handler_s, 0.0),
                count_name=f"{command}_calls",
            )
            self.add_client_timing("ipc_send_s", send_elapsed)
            self.add_client_timing("ipc_wait_s", wait_elapsed)
            self.add_client_timing("policy_queue_wait_s", queue_wait_s)
            self.add_client_timing("policy_handler_s", handler_s)
        return response.get("result")

    def reset_model(self) -> None:
        self.call("reset_model", track_timing=False)
        self.sim_seconds = 0.0
        self._reset_client_timing()

    def get_timing_rollout(self) -> dict[str, float | int]:
        timing = self.call("get_timing_rollout", track_timing=False)
        result = dict(timing or {})
        result["sim_s"] = float(self.sim_seconds)
        result.update(self._client_timing)
        return result

    def should_request_observation(self) -> bool:
        return True

    def close(self) -> None:
        try:
            if self.sock is not None:
                with self.profile_section("setup/ipc_close"):
                    self.sock.close()
        finally:
            self.sock = None  # type: ignore[assignment]
