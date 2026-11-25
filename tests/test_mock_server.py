"""Unit tests with mocked llama-server using httpx.MockTransport.

These tests verify the stillwarm core loop (restore -> completion -> save)
without requiring a real llama-server process or Docker environment.
"""

import json
from pathlib import Path

import httpx
import pytest

from stillwarm import LlamaServer, ServerProfile, Session
from stillwarm.settings import Settings


def make_mock_server() -> LlamaServer:
    """Create a mock LlamaServer that returns canned responses."""

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        method = request.method
        path = request.url.path
        query = request.url.query

        # Match health endpoint.
        if method == "GET" and path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        # Match save endpoint.
        if query and b"action=save" in query:
            return httpx.Response(
                200,
                json={"filename": "test.bin", "n_saved": 100, "n_written": 1234},
            )

        # Match restore endpoint.
        if query and b"action=restore" in query:
            return httpx.Response(200, json={"filename": "test.bin", "n_loaded": 100})

        # Match erase endpoint.
        if query and b"action=erase" in query:
            return httpx.Response(200, json={})

        # Match completion endpoint.
        if method == "POST" and path == "/completion":
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

        # Match tokenize endpoint.
        if method == "POST" and path == "/tokenize":
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
