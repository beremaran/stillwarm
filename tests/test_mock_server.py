"""Unit tests with mocked llama-server using httpx.MockTransport.

These tests verify the stillwarm core loop (restore -> completion -> save)
without requiring a real llama-server process or Docker environment.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest

from stillwarm import (
    PROBE_N_PREDICT,
    LlamaServer,
    ServerError,
    ServerProfile,
    Session,
    chat_probe_body,
)
from stillwarm.settings import Settings


def make_mock_server(record: dict[str, Any] | None = None) -> LlamaServer:  # noqa: C901
    """Create a mock LlamaServer that returns canned responses.

    If `record` is given, it collects:
      record["calls"]         -- endpoint names in request order
      record["chat_requests"] -- parsed bodies of /v1/chat/completions requests
    Setting record["chat_error"] to an HTTP status makes the chat endpoint fail.
    """

    def note(name: str) -> None:
        if record is not None:
            record.setdefault("calls", []).append(name)

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        method = request.method
        path = request.url.path
        query = request.url.query

        # Match health endpoint.
        if method == "GET" and path == "/health":
            note("health")
            return httpx.Response(200, json={"status": "ok"})

        # Match save endpoint.
        if query and b"action=save" in query:
            note("slots:save")
            return httpx.Response(
                200,
                json={"filename": "test.bin", "n_saved": 100, "n_written": 1234},
            )

        # Match restore endpoint.
        if query and b"action=restore" in query:
            note("slots:restore")
            return httpx.Response(200, json={"filename": "test.bin", "n_loaded": 100})

        # Match erase endpoint.
        if query and b"action=erase" in query:
            note("slots:erase")
            return httpx.Response(200, json={})

        # Match completion endpoint.
        if method == "POST" and path == "/completion":
            note("completion")
            body = json.loads(request.content.decode())
            prompt = body.get("prompt", "")
            n_predict = body.get("n_predict", 16)
            # Return a canned response.
            return httpx.Response(
                200,
                json={
                    "content": f"Mock response to: {prompt[:20]}...",
                    "tokens": [1, 2, 3],
                    "timings": {
                        "prompt_n": len(prompt),
                        "predicted_n": n_predict,
                        "prompt_ms": 100.0,
                        "predicted_ms": 500.0,
                    },
                },
            )

        # Match the OpenAI chat endpoint (non-stream).
        if method == "POST" and path == "/v1/chat/completions":
            note("chat")
            body = json.loads(request.content.decode())
            if record is not None:
                record.setdefault("chat_requests", []).append(body)
                chat_error = record.get("chat_error")
                if isinstance(chat_error, int):
                    return httpx.Response(chat_error, json={"error": "mock chat failure"})
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mock",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "mock probe reply"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 64, "total_tokens": 164},
                },
            )

        # Match tokenize endpoint.
        if method == "POST" and path == "/tokenize":
            note("tokenize")
            body = json.loads(request.content.decode())
            content = body.get("content", "")
            # Mock tokenizer: return deterministic token IDs.
            tokens = [ord(c) for c in content[:10]]
            return httpx.Response(200, json={"tokens": tokens})

        # Unknown endpoint.
        return httpx.Response(404, json={"error": f"unknown endpoint {method} {path}"})

    transport = httpx.MockTransport(handler)
    return LlamaServer("http://mock-server", transport=transport)


def test_session_save_and_restore(tmp_path: Path) -> None:
    """Verify the save/restore cycle works with a mock server."""
    profile = ServerProfile(
        model_sha256="test-sha256",
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        ctx_size=1024,
    )

    server = make_mock_server()
    sw = Session(
        server,
        profile,
        str(tmp_path),
        server_build_tag="test-build",
    )

    # Save a session.
    sw._live_prefix["test-session"] = "Test prefix text."  # noqa: SLF001
    sidecar = sw.save("test-session")

    # Verify sidecar was written.
    assert sidecar.model_sha256 == "test-sha256"
    assert sidecar.flash_attn == "on"
    assert sidecar.n_saved_tokens == 100

    # Restore the session.
    sw.server.slot_erase()
    verified = sw.restore("test-session", verify=False)
    assert verified is None  # No verify, so no verification result.


def test_session_ask_flow(tmp_path: Path) -> None:
    """Verify the full ask() flow with a mock server."""
    profile = ServerProfile(
        model_sha256="test-sha256",
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        ctx_size=1024,
    )

    server = make_mock_server()
    sw = Session(
        server,
        profile,
        str(tmp_path),
        server_build_tag="test-build",
    )

    # First ask: no restore, just save.
    result1 = sw.ask("test", "Test prompt.", n_predict=8)
    assert not result1.restored
    assert result1.content
    assert "Mock response" in result1.content

    # Verify sidecar was created.
    sidecar_file = tmp_path / "test.bin.stillwarm.json"
    assert sidecar_file.exists()

    # Verify prefix was saved.
    prefix_file = tmp_path / "test.prefix.txt"
    assert prefix_file.exists()


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Settings loads values from environment variables."""
    monkeypatch.setenv("STILLWARM_GUARD_RATIO", "0.7")
    monkeypatch.setenv("STILLWARM_GUARD_MIN_TOKENS", "128")

    settings = Settings()
    assert settings.guard_ratio == 0.7
    assert settings.guard_min_tokens == 128


def test_settings_from_env_file(tmp_path: Path) -> None:
    """Verify Settings loads values from a .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("STILLWARM_GUARD_RATIO=0.8\nSTILLWARM_GUARD_MIN_TOKENS=256\n")

    # Create settings instance pointing at the tmp_path .env file.
    class TestSettings(Settings):
        model_config = {  # noqa: RUF012
            "env_prefix": "STILLWARM_",
            "env_file": str(env_file),
            "extra": "ignore",
        }

    settings = TestSettings()
    assert settings.guard_ratio == 0.8
    assert settings.guard_min_tokens == 256


# --- Chat probe / save_chat (the proxy path) ---------------------------------


def _chat_profile() -> ServerProfile:
    return ServerProfile(
        model_sha256="test-sha256",
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        ctx_size=1024,
    )


def _chat_session(server: LlamaServer, cache_dir: str) -> Session:
    return Session(server, _chat_profile(), cache_dir, server_build_tag="test-build")


def _sample_chat_body() -> dict[str, Any]:
    return {
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "Hello?"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "seed": 1234,
        "stream": True,
        "stream_options": {"include_usage": True},
        "stop": ["\n\n"],
    }


def test_chat_probe_body_pins_sampler() -> None:
    """The probe pins greedy generation and passes prompt-affecting fields verbatim."""
    original = _sample_chat_body()
    probe = chat_probe_body(original)

    assert probe["temperature"] == 0
    assert probe["top_k"] == 1
    assert probe["ignore_eos"] is True
    assert probe["max_tokens"] == PROBE_N_PREDICT
    assert probe["stream"] is False

    for stripped in ("top_p", "seed", "stream_options"):
        assert stripped not in probe

    assert probe["messages"] == original["messages"]
    assert probe["tools"] == original["tools"]
    assert probe["model"] == original["model"]
    assert probe["stop"] == original["stop"]


def test_save_chat_writes_probed_triple(tmp_path: Path) -> None:
    """save_chat pins the probe, hashes the message, saves before probing."""
    record: dict[str, Any] = {}
    sw = _chat_session(make_mock_server(record), str(tmp_path))

    original = _sample_chat_body()
    sidecar = sw.save_chat("chat-sess", original, generated_tokens=13)

    # Invariant: slot_save runs before the probe.
    calls = record["calls"]
    assert calls.index("slots:save") < calls.index("chat")

    # The probe request is the pinned, non-streaming form of the original.
    probe_request = record["chat_requests"][0]
    assert probe_request["temperature"] == 0
    assert probe_request["top_k"] == 1
    assert probe_request["ignore_eos"] is True
    assert probe_request["max_tokens"] == PROBE_N_PREDICT
    assert probe_request["stream"] is False
    assert probe_request["messages"] == original["messages"]
    assert probe_request["id_slot"] == 0

    # The probe hash is the canonical JSON of the generated message.
    expected_hash = hashlib.sha256(
        orjson.dumps(
            {"content": "mock probe reply", "role": "assistant"},
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()
    assert sidecar.probe_hash == expected_hash
    assert sidecar.probe_n_predict == PROBE_N_PREDICT
    assert json.loads(sidecar.probe_sampler_json) == {
        "temperature": 0,
        "top_k": 1,
        "ignore_eos": True,
        "max_tokens": PROBE_N_PREDICT,
    }
    assert sidecar.save_point == "after_generation"
    assert sidecar.n_saved_tokens == 100

    # .prefix.txt holds the replayable probe record; the sidecar hashes it.
    prefix = (tmp_path / "chat-sess.prefix.txt").read_bytes()
    assert json.loads(prefix) == probe_request
    assert sidecar.prompt_prefix_sha256 == hashlib.sha256(prefix).hexdigest()


def test_save_chat_keeps_client_id_slot(tmp_path: Path) -> None:
    """A client-specified id_slot in the request wins over the session default."""
    record: dict[str, Any] = {}
    sw = _chat_session(make_mock_server(record), str(tmp_path))

    original = _sample_chat_body()
    original["id_slot"] = 3
    sw.save_chat("slot3", original)

    assert record["chat_requests"][0]["id_slot"] == 3


def test_save_chat_without_probe(tmp_path: Path) -> None:
    """probe_body=None writes an unprobed sidecar and an empty probe record."""
    record: dict[str, Any] = {}
    sw = _chat_session(make_mock_server(record), str(tmp_path))

    sidecar = sw.save_chat("unprobed", None)

    assert sidecar.probe_hash == ""
    assert sidecar.probe_n_predict == 0
    assert sidecar.probe_sampler_json == ""
    assert sidecar.save_point == "doc_prefill"
    assert "chat" not in record["calls"]
    assert (tmp_path / "unprobed.prefix.txt").read_bytes() == b""
    assert sidecar.prompt_prefix_sha256 == hashlib.sha256(b"").hexdigest()


def test_restore_verify_empty_probe_short_circuits(tmp_path: Path) -> None:
    """restore(verify=True) on an unprobed sidecar sends no probe and returns None."""
    record: dict[str, Any] = {}
    sw = _chat_session(make_mock_server(record), str(tmp_path))
    sw.save_chat("unprobed", None)

    verified = sw.restore("unprobed", verify=True)

    assert verified is None
    assert "chat" not in record["calls"]
    assert "completion" not in record["calls"]


def test_save_chat_probe_failure_deletes_triple(tmp_path: Path) -> None:
    """A failed probe removes the half-written triple and re-raises."""
    record: dict[str, Any] = {"chat_error": 500}
    sw = _chat_session(make_mock_server(record), str(tmp_path))
    (tmp_path / "csess.bin").write_bytes(b"kv")  # the real server writes this

    with pytest.raises(ServerError):
        sw.save_chat("csess", _sample_chat_body(), generated_tokens=4)

    assert not (tmp_path / "csess.bin").exists()
    assert not (tmp_path / "csess.bin.stillwarm.json").exists()
    assert not (tmp_path / "csess.prefix.txt").exists()
