"""Proxy middleware tests: the chat loop, identity, auth, pass-through, tee.

The Starlette app runs in-process over httpx.ASGITransport. The "upstream
llama-server" is played twice, the way it really is one process: the sync
LlamaServer (slot ops + probe) and the async upstream client (the client's
request) hit the same server, each with its own record.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, MutableMapping
from pathlib import Path
from typing import Any, override

import httpx
import orjson
import pytest
from starlette.applications import Starlette
from test_mock_server import make_mock_server

from stillwarm.proxy import (
    ChatTee,
    ProxyConfig,
    ProxyStartupError,
    build_app,
    derived_session_id,
)
from stillwarm.server import LlamaServer, ServerError
from stillwarm.sidecar import Sidecar

CHAT_URL = "/v1/chat/completions"
JSON_HEADERS = {"content-type": "application/json"}

SYSTEM = {"role": "system", "content": "You are a test."}
USER1 = {"role": "user", "content": "Hello?"}
TURN1_BODY: dict[str, Any] = {
    "model": "mock",
    "messages": [SYSTEM, USER1],
    "temperature": 0.7,
    "max_tokens": 32,
}
TURN1_BYTES = orjson.dumps(TURN1_BODY)
TURN2_BODY: dict[str, Any] = {
    "model": "mock",
    "messages": [
        SYSTEM,
        USER1,
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "And you?"},
    ],
    "temperature": 0.7,
    "max_tokens": 32,
}


def expected_derived_id() -> str:
    """The id the proxy must derive for every turn of this conversation."""
    canonical = orjson.dumps([SYSTEM, USER1], option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


def _cfg(tmp_path: Path, **overrides: Any) -> ProxyConfig:
    base: dict[str, Any] = {
        "base_url": "http://upstream",
        "cache_dir": str(tmp_path / "saves"),
        "model_sha256": "m" * 64,
        "flash_attn": "on",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "swa_full": False,
        "ctx_size": 4096,
        "host": "127.0.0.1",
        "port": 8081,
        "token": "",
        "timeout": 5.0,
        "disk_budget_gb": None,
        "no_probe": False,
    }
    base.update(overrides)
    return ProxyConfig(**base)


class _UpstreamStream(httpx.AsyncByteStream):
    """A one-shot body stream, so mock responses behave like a live connection.

    SSE bodies arrive one frame (blank line) at a time, with a yield to the
    event loop between frames, the way a real streaming server delivers.
    Without the yields the proxy would relay the whole body in one go and a
    mid-stream disconnect could never interleave.
    """

    def __init__(self, data: bytes) -> None:
        sep = b"\r\n\r\n" if b"\r\n\r\n" in data else b"\n\n"
        parts = data.split(sep)
        self._chunks = [part + sep for part in parts[:-1]]
        if parts[-1]:
            self._chunks.append(parts[-1])

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, chunk in enumerate(self._chunks):
            if i:
                await asyncio.sleep(0)
            yield chunk


def _upstream_response(status: int, body: bytes, content_type: str) -> httpx.Response:
    """Build a mock upstream response whose body is still streaming.

    ``httpx.Response(content=...)`` eagerly reads the body, which would make
    the proxy's ``aiter_raw()`` raise ``StreamConsumed``.
    """
    return httpx.Response(
        status, headers={"content-type": content_type}, stream=_UpstreamStream(body)
    )


def _make_upstream(  # noqa: PLR0913 — one knob per upstream behavior under test
    calls: list[tuple[str, str, bytes, dict[str, str]]],
    *,
    sse: bytes | None = None,
    chat_json: dict[str, Any] | None = None,
    chat_status: int = 200,
    chat_cached: int | None = 100,
    chat_error: str | None = None,
) -> httpx.AsyncClient:
    """The client-facing side of the same (mock) llama-server."""

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            (
                request.method,
                request.url.path,
                request.content,
                dict(request.headers.multi_items()),
            )
        )
        path = request.url.path
        if path == "/props":
            return _upstream_response(
                200,
                orjson.dumps({"build_info": "b9871", "total_slots": 1, "n_ctx": 4096}),
                "application/json",
            )
        if path == CHAT_URL:
            if chat_error is not None:
                raise httpx.ConnectError(chat_error)
            body: dict[str, Any]
            try:
                body = orjson.loads(request.content)
            except ValueError:
                body = {}
            if body.get("stream"):
                return _upstream_response(200, sse or _sse_bytes(), "text/event-stream")
            if chat_status != 200:
                return _upstream_response(
                    chat_status, orjson.dumps({"error": "upstream failed"}), "application/json"
                )
            return _upstream_response(
                200, orjson.dumps(chat_json or _chat_json(cached=chat_cached)), "application/json"
            )
        return _upstream_response(
            200,
            orjson.dumps({"echo": path, "method": request.method}),
            "application/json",
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chat_json(
    *, cached: int | None = 100, completion: int = 7, prompt: int = 105
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "timings": {"prompt_n": prompt, "predicted_n": completion, "predicted_ms": 10.0},
    }


def _sse_bytes() -> bytes:
    frames: list[str | dict[str, Any]] = [
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 105,
                "completion_tokens": 7,
                "total_tokens": 112,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
            "timings": {"prompt_n": 105, "predicted_n": 7},
        },
        "[DONE]",
    ]
    return b"".join(
        b"data: " + (b"[DONE]" if isinstance(f, str) else orjson.dumps(f)) + b"\n\n" for f in frames
    )


def _client(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


def _build(
    tmp_path: Path,
    sync: dict[str, Any],
    calls: list[tuple[str, str, bytes, dict[str, str]]],
    *,
    cfg: dict[str, Any] | None = None,
    upstream: dict[str, Any] | None = None,
) -> Starlette:
    server = make_mock_server(sync)
    up = _make_upstream(calls, **(upstream or {}))
    return build_app(_cfg(tmp_path, **(cfg or {})), server=server, upstream=up)


# --- the chat loop -----------------------------------------------------------


def test_first_turn_cold_writes_triple(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = expected_derived_id()

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
        assert resp.status_code == 200
        assert resp.headers["x-stillwarm-restore"] == "cold"
        # the client's body is forwarded byte-for-byte, unmodified
        method, path, body, headers = calls[0]
        assert (method, path) == ("POST", CHAT_URL)
        assert body == TURN1_BYTES
        assert "content-type" in headers
        # slot_save runs before the probe
        assert sync["calls"].index("slots:save") < sync["calls"].index("chat")
        # the probe is the pinned form of the original request
        probe = sync["chat_requests"][0]
        assert probe["temperature"] == 0
        assert probe["top_k"] == 1
        assert probe["ignore_eos"] is True
        assert probe["stream"] is False
        assert probe["id_slot"] == 0
        assert probe["messages"] == TURN1_BODY["messages"]
        # the triple is on disk under the derived id
        saves = tmp_path / "saves"
        assert (saves / f"{sid}.bin").read_bytes() == b"\0" * 1234
        sc = Sidecar.read(saves / f"{sid}.bin.stillwarm.json")
        assert sc.save_point == "after_generation"
        assert sc.probe_hash
        prefix = (saves / f"{sid}.prefix.txt").read_bytes()
        assert json.loads(prefix) == probe

    asyncio.run(scenario())


def test_second_turn_ok_erases_and_restores(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = expected_derived_id()

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            r1 = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
            r2 = await client.post(CHAT_URL, content=orjson.dumps(TURN2_BODY), headers=JSON_HEADERS)
        assert r1.headers["x-stillwarm-restore"] == "cold"
        assert r2.headers["x-stillwarm-restore"] == "ok"
        # turn 2 re-derives the same id from the resent history
        assert (tmp_path / "saves" / f"{sid}.bin").exists()
        assert calls[1][2] == orjson.dumps(TURN2_BODY)
        # erase+restore precede the forward; save+probe follow it
        assert sync["calls"] == [
            "props",
            "slots:save",
            "chat",
            "slots:erase",
            "slots:restore",
            "slots:save",
            "chat",
        ]

    asyncio.run(scenario())


def test_restore_refused_on_model_mismatch(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = "refused-sess"

    async def scenario() -> None:
        saves = tmp_path / "saves"
        saves.mkdir(parents=True)
        (saves / f"{sid}.bin").write_bytes(b"\0" * 100)
        Sidecar(
            model_sha256="x" * 64,  # saved for a different model
            flash_attn="on",
            cache_type_k="f16",
            cache_type_v="f16",
            swa_full=False,
            n_saved_tokens=100,
            prompt_prefix_sha256="p" * 64,
            save_file=f"{sid}.bin",
        ).write(saves / f"{sid}.bin.stillwarm.json")
        (saves / f"{sid}.prefix.txt").write_bytes(b"{}")
        sync["save_dir"] = str(saves)
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL,
                content=TURN1_BYTES,
                headers={"X-Stillwarm-Session": sid, **JSON_HEADERS},
            )
        assert resp.status_code == 200
        assert resp.headers["x-stillwarm-restore"] == "refused"
        # the slot was erased but never restored; the request went out cold
        assert sync["calls"] == ["props", "slots:erase", "slots:save", "chat"]
        # the refused triple is replaced by a save for the current model
        sc = Sidecar.read(saves / f"{sid}.bin.stillwarm.json")
        assert sc.model_sha256 == "m" * 64

    asyncio.run(scenario())


def test_orphan_bin_without_sidecar_starts_cold(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = "orphan-sess"

    async def scenario() -> None:
        saves = tmp_path / "saves"
        saves.mkdir(parents=True)
        (saves / f"{sid}.bin").write_bytes(b"\0" * 100)  # no sidecar: not a session
        sync["save_dir"] = str(saves)
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL,
                content=TURN1_BYTES,
                headers={"X-Stillwarm-Session": sid, **JSON_HEADERS},
            )
        assert resp.status_code == 200
        assert resp.headers["x-stillwarm-restore"] == "cold"
        # the orphan was removed; a real triple was written for the new state
        assert (saves / f"{sid}.bin.stillwarm.json").exists()
        assert "slots:restore" not in sync["calls"]
        assert "slots:erase" in sync["calls"]

    asyncio.run(scenario())


def test_header_id_becomes_filename(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            resp = await client.post(
                CHAT_URL,
                content=TURN1_BYTES,
                headers={"X-Stillwarm-Session": "my.sess-1", **JSON_HEADERS},
            )
        assert resp.status_code == 200
        saves = tmp_path / "saves"
        assert (saves / "my.sess-1.bin").exists()
        assert not (saves / f"{expected_derived_id()}.bin").exists()

    asyncio.run(scenario())


def test_bad_session_header_400(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            for bad in ("../etc/passwd", "a" * 65, "has space", ""):
                resp = await client.post(
                    CHAT_URL,
                    content=TURN1_BYTES,
                    headers={"X-Stillwarm-Session": bad, **JSON_HEADERS},
                )
                assert resp.status_code == 400
        # nothing reached the upstream, no slot touched
        assert calls == []
        assert sync["calls"] == ["props"]

    asyncio.run(scenario())


def test_derived_id_stable_and_sensitive() -> None:
    grown = {"messages": [SYSTEM, USER1, {"role": "user", "content": "more"}]}
    base = {"messages": [SYSTEM, USER1]}
    assert derived_session_id(grown) == derived_session_id(base)
    new_system = {"messages": [{"role": "system", "content": "Different"}, USER1]}
    assert derived_session_id(new_system) != derived_session_id(base)
    new_user = {"messages": [SYSTEM, {"role": "user", "content": "Other"}]}
    assert derived_session_id(new_user) != derived_session_id(base)
    assert derived_session_id({"messages": []}) == derived_session_id({})


# --- pass-through ------------------------------------------------------------


def test_passthrough_paths_verbatim(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    completion_body = b'{"prompt": "hi", "n_predict": 1}'

    async def scenario() -> None:
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            props = await client.get("/props")
            comp = await client.post("/completion", content=completion_body, headers=JSON_HEADERS)
        assert props.status_code == 200
        assert props.json() == {
            "build_info": "b9871",
            "total_slots": 1,
            "n_ctx": 4096,
        }
        assert "x-stillwarm-restore" not in props.headers
        assert comp.status_code == 200
        assert comp.json()["echo"] == "/completion"
        assert calls[1][2] == completion_body
        # only the startup props went through the library client
        assert sync["calls"] == ["props"]
        assert calls[0][0] == "GET"

    asyncio.run(scenario())


def test_chat_path_invalid_json_passthrough(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        app = _build(tmp_path, sync, calls)
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=b"not json", headers=JSON_HEADERS)
        assert resp.status_code == 200
        assert calls[0][2] == b"not json"
        assert "x-stillwarm-restore" not in resp.headers
        assert sync["calls"] == ["props"]

    asyncio.run(scenario())


def test_bearer_token_enforced_and_never_forwarded(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        app = _build(tmp_path, sync, calls, cfg={"token": "sekret"})
        async with _client(app) as client:
            denied = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
            wrong = await client.post(
                CHAT_URL,
                content=TURN1_BYTES,
                headers={"authorization": "Bearer no", **JSON_HEADERS},
            )
            ok = await client.post(
                CHAT_URL,
                content=TURN1_BYTES,
                headers={"authorization": "Bearer sekret", **JSON_HEADERS},
            )
        assert denied.status_code == 401
        assert wrong.status_code == 401
        assert ok.status_code == 200
        for _, _, _, headers in calls:
            assert "authorization" not in headers

    asyncio.run(scenario())


# --- responses: save signals, tee, probe off, errors --------------------------


def test_sse_stream_relayed_byte_identical_and_saved(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = expected_derived_id()

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls)
        body = orjson.dumps(
            {**TURN1_BODY, "stream": True, "stream_options": {"include_usage": True}}
        )
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=body, headers=JSON_HEADERS)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream"
        assert resp.content == _sse_bytes()
        assert resp.headers["x-stillwarm-restore"] == "cold"
        saves = tmp_path / "saves"
        sc = Sidecar.read(saves / f"{sid}.bin.stillwarm.json")
        assert sc.save_point == "after_generation"
        # the probe strips the streaming fields and pins generation
        probe = sync["chat_requests"][0]
        assert probe["stream"] is False
        assert "stream_options" not in probe
        assert probe["max_tokens"] == 64
        assert probe["messages"] == TURN1_BODY["messages"]

    asyncio.run(scenario())


def test_tool_call_delta_triggers_save(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = expected_derived_id()
    frames: list[str | dict[str, Any]] = [
        {
            "id": "c",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    ]
    sse = b"".join(
        b"data: " + (b"[DONE]" if isinstance(f, str) else orjson.dumps(f)) + b"\n\n" for f in frames
    )

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls, upstream={"sse": sse})
        body = orjson.dumps({**TURN1_BODY, "stream": True})
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=body, headers=JSON_HEADERS)
        assert resp.status_code == 200
        assert resp.content == sse
        # no usage in the stream: the tool-call delta is the save signal
        saves = tmp_path / "saves"
        sc = Sidecar.read(saves / f"{sid}.bin.stillwarm.json")
        assert sc.save_point == "after_generation"

    asyncio.run(scenario())


def test_no_generation_no_save(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    bare: dict[str, Any] = {
        "id": "c",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"}
        ],
    }

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls, upstream={"chat_json": bare})
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
        assert resp.status_code == 200
        assert json.loads(resp.content)["choices"][0]["message"]["content"] is None
        # no usage, no delta: nothing worth saving
        assert sync["calls"] == ["props"]

    asyncio.run(scenario())


def test_no_probe_flag_skips_probe(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    sid = expected_derived_id()

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls, cfg={"no_probe": True})
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
        assert resp.status_code == 200
        saves = tmp_path / "saves"
        sc = Sidecar.read(saves / f"{sid}.bin.stillwarm.json")
        assert sc.probe_hash == ""
        assert sc.probe_n_predict == 0
        assert (saves / f"{sid}.prefix.txt").read_bytes() == b""
        assert "chat" not in sync["calls"]

    asyncio.run(scenario())


def test_guard_warns_when_cache_not_reused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls, upstream={"chat_cached": 0})
        async with _client(app) as client:
            await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
            await client.post(CHAT_URL, content=orjson.dumps(TURN2_BODY), headers=JSON_HEADERS)

    with caplog.at_level(logging.WARNING, logger="stillwarm.proxy"):
        asyncio.run(scenario())
    assert "SILENTLY INEFFECTIVE" in caplog.text


def test_prune_after_save_when_budget_set(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        saves = tmp_path / "saves"
        saves.mkdir(parents=True)
        old = saves / "old-sess.bin"
        old.write_bytes(b"\0" * 600)
        Sidecar(
            model_sha256="m" * 64,
            flash_attn="on",
            cache_type_k="f16",
            cache_type_v="f16",
            swa_full=False,
            n_saved_tokens=10,
            prompt_prefix_sha256="p" * 64,
            save_file=old.name,
        ).write(saves / "old-sess.bin.stillwarm.json")
        (saves / "old-sess.prefix.txt").write_text("old")
        t = time.time() - 3600
        os.utime(old, (t, t))
        sync["save_dir"] = str(saves)
        app = _build(tmp_path, sync, calls, cfg={"disk_budget_gb": 1500 / 1e9})
        async with _client(app) as client:
            await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)

    asyncio.run(scenario())
    saves = tmp_path / "saves"
    assert not (saves / "old-sess.bin").exists()
    assert not (saves / "old-sess.bin.stillwarm.json").exists()
    assert not (saves / "old-sess.prefix.txt").exists()
    assert (saves / f"{expected_derived_id()}.bin").exists()


def test_upstream_unreachable_502_and_lock_released(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        app = _build(tmp_path, sync, calls, upstream={"chat_error": "boom"})
        async with _client(app) as client:
            denied = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
            recover = await client.get("/props")
        assert denied.status_code == 502
        assert "upstream" in denied.text
        assert recover.status_code == 200  # the lock was released

    asyncio.run(scenario())


def test_upstream_http_error_forwarded_without_save(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls, upstream={"chat_status": 500})
        async with _client(app) as client:
            resp = await client.post(CHAT_URL, content=TURN1_BYTES, headers=JSON_HEADERS)
        assert resp.status_code == 500
        assert json.loads(resp.content) == {"error": "upstream failed"}
        assert resp.headers["x-stillwarm-restore"] == "cold"
        assert sync["calls"] == ["props"]  # no save on a failed request

    asyncio.run(scenario())


def test_mid_stream_disconnect_no_save(tmp_path: Path) -> None:
    """A mid-stream client abort skips the save and releases the lock.

    httpx.ASGITransport runs the app to completion before the client reads a
    byte, so a disconnect cannot interleave with it. This test drives the
    ASGI app directly and issues http.disconnect after the first frame, the
    way uvicorn would.
    """
    sync: dict[str, Any] = {}
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def scenario() -> None:
        sync["save_dir"] = str(tmp_path / "saves")
        app = _build(tmp_path, sync, calls)
        body = orjson.dumps({**TURN1_BODY, "stream": True})
        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "headers": [(b"content-type", b"application/json"), (b"host", b"proxy")],
            "scheme": "http",
            "path": CHAT_URL,
            "raw_path": CHAT_URL.encode("ascii"),
            "query_string": b"",
            "server": ("proxy", 8081),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }
        first_chunk = asyncio.Event()
        disconnect = asyncio.Event()
        stage = 0

        async def receive() -> dict[str, Any]:
            nonlocal stage
            if stage == 0:
                stage = 1
                return {"type": "http.request", "body": body, "more_body": True}
            if stage == 1:
                stage = 2
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.body" and message.get("body"):
                first_chunk.set()

        app_task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(first_chunk.wait(), timeout=5)
        disconnect.set()  # the client abandons the connection
        await asyncio.wait_for(app_task, timeout=5)
        assert not app.state.proxy.lock.locked()  # the abort released the lock

    asyncio.run(scenario())
    assert not list((tmp_path / "saves").glob("*.stillwarm.json"))
    assert "slots:save" not in sync["calls"]


# --- startup ------------------------------------------------------------------


def test_build_app_rejects_unresolved_flash_attn(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    server = make_mock_server(sync)
    with pytest.raises(ProxyStartupError, match="flash_attn"):
        build_app(
            _cfg(tmp_path, flash_attn="auto"),
            server=server,
            upstream=httpx.AsyncClient(),
        )
    assert sync.get("calls", []) == []  # fails before touching the upstream


def test_build_app_requires_reachable_props(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "nope"})

    server = LlamaServer("http://upstream", transport=httpx.MockTransport(handler))
    with pytest.raises(ServerError):
        build_app(_cfg(tmp_path), server=server, upstream=httpx.AsyncClient())


def test_build_app_takes_ctx_and_build_tag_from_props(tmp_path: Path) -> None:
    sync: dict[str, Any] = {}
    server = make_mock_server(sync)
    app = build_app(_cfg(tmp_path, ctx_size=0), server=server, upstream=httpx.AsyncClient())
    proxy = app.state.proxy
    assert proxy.session.profile.ctx_size == 4096
    assert proxy.session.server_build_tag == "b9871"


def test_build_app_requires_ctx_size(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"build_info": "b9871", "total_slots": 1})
        return httpx.Response(404, json={})

    server = LlamaServer("http://upstream", transport=httpx.MockTransport(handler))
    with pytest.raises(ProxyStartupError, match="ctx"):
        build_app(_cfg(tmp_path, ctx_size=0), server=server, upstream=httpx.AsyncClient())


def test_build_app_warns_on_multi_slot(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"build_info": "b9871", "total_slots": 2, "n_ctx": 4096}
            )
        return httpx.Response(404, json={})

    server = LlamaServer("http://upstream", transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="stillwarm.proxy"):
        build_app(_cfg(tmp_path), server=server, upstream=httpx.AsyncClient())
    assert "slots" in caplog.text


# --- ChatTee unit tests --------------------------------------------------------


def test_tee_handles_split_frames() -> None:
    sse = _sse_bytes()
    tee = ChatTee(sse=True)
    relayed = b""
    for i in range(len(sse)):  # one byte at a time: worst-case chunking
        relayed += b"".join(tee.feed(sse[i : i + 1]))
    relayed += b"".join(tee.finish())
    assert relayed == sse
    assert tee.saw_generation
    assert tee.generated_tokens() == 7
    assert tee.cached_tokens() == 100
    assert tee.timings["prompt_n"] == 105
    assert tee.has_save_point()


def test_tee_crlf_frames() -> None:
    payload = orjson.dumps(
        {
            "choices": [{"delta": {"content": "x"}}],
            "usage": {
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }
    )
    sse = b"data: " + payload + b"\r\n\r\ndata: [DONE]\r\n\r\n"
    tee = ChatTee(sse=True)
    relayed = b"".join(tee.feed(sse[:3]))
    relayed += b"".join(tee.feed(sse[3:10]))
    relayed += b"".join(tee.feed(sse[10:]))
    relayed += b"".join(tee.finish())
    assert relayed == sse
    assert tee.generated_tokens() == 3
    assert tee.cached_tokens() == 2


def test_tee_non_stream_parses_whole_body() -> None:
    body = orjson.dumps(_chat_json())
    tee = ChatTee(sse=False)
    relayed = b"".join(tee.feed(body[:10]))
    relayed += b"".join(tee.feed(body[10:]))
    assert relayed == body
    assert tee.finish() == []
    assert tee.generated_tokens() == 7
    assert tee.cached_tokens() == 100


def test_tee_malformed_chunk_is_skipped() -> None:
    sse = b"data: {not json}\n\ndata: [DONE]\n\n"
    tee = ChatTee(sse=True)
    relayed = b"".join(tee.feed(sse))
    relayed += b"".join(tee.finish())
    assert relayed == sse
    assert not tee.has_save_point()
