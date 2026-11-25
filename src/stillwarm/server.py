# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""Thin llama-server HTTP client. Standard library only (zero-dep contract).

Covers exactly the endpoints stillwarm needs, with shapes verified against the
pinned llama.cpp build (b9871) during the stillwarm-bench measurements:
  GET  /health
  GET  /props
  POST /tokenize            {"content": str}            -> {"tokens": [int]}
  POST /completion          (non-stream)                -> {..., "timings": {...}}
  POST /slots/{id}?action=save|restore|erase  {"filename": str}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast


class ServerError(RuntimeError):
    """An HTTP error from llama-server, with the response body preserved."""

    def __init__(self, status: int, body: str) -> None:
        """Initialize with HTTP status code and response body."""
        self.status = status
        self.body = body
        super().__init__(f"llama-server HTTP {status}: {body[:300]}")


PROBE_SAMPLER: dict[str, Any] = {"temperature": 0, "top_k": 1, "seed": 42, "ignore_eos": True}
PROBE_N_PREDICT = 64
DEFAULT_SAMPLER: dict[str, Any] = {"temperature": 0, "top_k": 1, "seed": 42}


class LlamaServer:
    """HTTP client for llama-server slot save/restore endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 600.0) -> None:
        """Initialize with server base URL and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = urllib.parse.urljoin(self.base_url, path)
        scheme = urllib.parse.urlparse(url).scheme
        if scheme not in ("http", "https"):
            msg = f"URL must use http or https scheme: {url}"
            raise ValueError(msg)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return dict(json.loads(resp.read().decode("utf-8") or "{}"))
        except urllib.error.HTTPError as e:
            raise ServerError(e.code, e.read().decode("utf-8", "replace")) from None

    def health(self) -> bool:
        """Check if the server is responding."""
        try:
            self._request("GET", "/health")
        except (ServerError, urllib.error.URLError):
            return False
        else:
            return True

    def props(self) -> dict[str, Any]:
        """Get server properties including loaded model info."""
        return self._request("GET", "/props")

    def ctx_size(self) -> int | None:
        """Per-slot context size, from /props when the build exposes it."""
        props = self.props()
        for probe in (("default_generation_settings", "n_ctx"),):
            v = props.get(probe[0], {})
            if isinstance(v, dict) and isinstance(v.get(probe[1]), int):
                return cast("int", v[probe[1]])
        return None

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text and return token IDs."""
        return cast("list[int]", self._request("POST", "/tokenize", {"content": text})["tokens"])

    def completion(  # noqa: PLR0913 — mirrors llama-server completion endpoint signature
        self,
        prompt: str,
        n_predict: int,
        *,
        cache_prompt: bool = True,
        id_slot: int = 0,
        return_tokens: bool = True,
        sampler: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a completion with the given prompt and parameters."""
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "id_slot": id_slot,
            "return_tokens": return_tokens,
            "stream": False,
        }
        body.update(sampler or PROBE_SAMPLER)
        return self._request("POST", "/completion", body)

    def slot_save(self, filename: str, id_slot: int = 0) -> dict[str, Any]:
        """Save the KV cache for a slot to disk."""
        return self._request("POST", f"/slots/{id_slot}?action=save", {"filename": filename})

    def slot_restore(self, filename: str, id_slot: int = 0) -> dict[str, Any]:
        """Restore the KV cache for a slot from disk."""
        return self._request("POST", f"/slots/{id_slot}?action=restore", {"filename": filename})

    def slot_erase(self, id_slot: int = 0) -> dict[str, Any]:
        """Erase the slot, clearing its KV cache."""
        return self._request("POST", f"/slots/{id_slot}?action=erase", {})
