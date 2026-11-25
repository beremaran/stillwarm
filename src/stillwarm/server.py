# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""Thin llama-server HTTP client.

Covers exactly the endpoints stillwarm needs, with shapes verified against
llama.cpp release b9871:
  GET  /health
  GET  /props
  POST /tokenize            {"content": str}            -> {"tokens": [int]}
  POST /completion          (non-stream)                -> {..., "timings": {...}}
  POST /v1/chat/completions (non-stream)                -> OpenAI chat response
  POST /slots/{id}?action=save|restore|erase  {"filename": str}
"""

from typing import Any, cast

import httpx


class ServerError(RuntimeError):
    """An HTTP error from llama-server, with the response body preserved."""

    def __init__(self, status: int, body: str) -> None:
        """Initialize with HTTP status code and response body."""
        self.status = status
        self.body = body
        super().__init__(f"llama-server HTTP {status}: {body[:_MAX_ERROR_BODY]}")


# Greedy sampling (temperature 0, top_k 1) is deterministic on its own, so no
# seed is needed or recorded: these samplers are fully specified by the other
# fields.
PROBE_SAMPLER: dict[str, Any] = {"temperature": 0, "top_k": 1, "ignore_eos": True}
PROBE_N_PREDICT = 64
DEFAULT_SAMPLER: dict[str, Any] = {"temperature": 0, "top_k": 1}

DEFAULT_TIMEOUT_S = 600.0
_MAX_ERROR_BODY = 300


class LlamaServer:
    """HTTP client for llama-server slot save/restore endpoints.

    Uses a persistent httpx.Client for connection pooling. For testing,
    pass a custom transport via the `transport` kwarg.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize with the server's base URL and request timeout.

        base_url may be any reachable http(s) URL, local or remote.
        transport, if provided, replaces the default HTTP transport.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=transport,
        )

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        resp = self._client.request(method, path, json=payload)
        if resp.is_error:
            raise ServerError(resp.status_code, resp.text)
        return dict(resp.json() or {})

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def health(self) -> bool:
        """Check if the server is responding."""
        try:
            self._request("GET", "/health")
        except (ServerError, httpx.RequestError):
            return False
        else:
            return True

    def props(self) -> dict[str, Any]:
        """Server properties: build_info, total_slots, model metadata."""
        return self._request("GET", "/props")

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

    def chat(
        self,
        body: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Non-streaming OpenAI chat completion against /v1/chat/completions.

        The body is passed through verbatim except that `stream` is forced to
        False: this method returns the parsed JSON response, so a stream would
        be meaningless. `timeout` overrides the client-wide timeout for this
        request alone (the save probe needs a short one). HTTP errors raise
        ServerError; transport errors propagate as httpx.RequestError.
        """
        payload = {**body, "stream": False}
        path = "/v1/chat/completions"
        try:
            resp = self._client.request("POST", path, json=payload, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ServerError(e.response.status_code, e.response.text) from None
        return dict(resp.json() or {})

    def slot_save(self, filename: str, id_slot: int = 0) -> dict[str, Any]:
        """Save the KV cache for a slot to disk."""
        return self._request("POST", f"/slots/{id_slot}?action=save", {"filename": filename})

    def slot_restore(self, filename: str, id_slot: int = 0) -> dict[str, Any]:
        """Restore the KV cache for a slot from disk."""
        return self._request("POST", f"/slots/{id_slot}?action=restore", {"filename": filename})

    def slot_erase(self, id_slot: int = 0) -> dict[str, Any]:
        """Erase the slot, clearing its KV cache."""
        return self._request("POST", f"/slots/{id_slot}?action=erase", {})
