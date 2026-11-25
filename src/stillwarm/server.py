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
import urllib.request
from typing import Any, Dict, List, Optional


class ServerError(RuntimeError):
    """An HTTP error from llama-server, with the response body preserved."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"llama-server HTTP {status}: {body[:300]}")


#: pinned sampler parameters for probe generation — part of the sidecar contract.
#: ignore_eos keeps every probe exactly PROBE_N_PREDICT tokens (constant-length
#: probes make the determinism comparison independent of where EOS falls).
PROBE_SAMPLER: Dict[str, Any] = {"temperature": 0, "top_k": 1, "seed": 42, "ignore_eos": True}
PROBE_N_PREDICT = 64

#: default sampler for user requests through Session.ask (greedy, EOS honored).
DEFAULT_SAMPLER: Dict[str, Any] = {"temperature": 0, "top_k": 1, "seed": 42}


class LlamaServer:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing ------------------------------------------------------------
    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise ServerError(e.code, e.read().decode("utf-8", "replace")) from None

    # -- endpoints -----------------------------------------------------------
    def health(self) -> bool:
        try:
            self._request("GET", "/health")
            return True
        except (ServerError, urllib.error.URLError):
            return False

    def props(self) -> dict:
        return self._request("GET", "/props")

    def ctx_size(self) -> Optional[int]:
        """Per-slot context size, from /props when the build exposes it."""
        props = self.props()
        for probe in (("default_generation_settings", "n_ctx"),):
            v = props.get(probe[0], {})
            if isinstance(v, dict) and isinstance(v.get(probe[1]), int):
                return v[probe[1]]
        return None

    def tokenize(self, text: str) -> List[int]:
        return self._request("POST", "/tokenize", {"content": text})["tokens"]

    def completion(
        self,
        prompt: str,
        n_predict: int,
        *,
        cache_prompt: bool = True,
        id_slot: int = 0,
        return_tokens: bool = True,
        sampler: Optional[dict] = None,
    ) -> dict:
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

    def slot_save(self, filename: str, id_slot: int = 0) -> dict:
        return self._request("POST", f"/slots/{id_slot}?action=save", {"filename": filename})

    def slot_restore(self, filename: str, id_slot: int = 0) -> dict:
        return self._request("POST", f"/slots/{id_slot}?action=restore", {"filename": filename})

    def slot_erase(self, id_slot: int = 0) -> dict:
        return self._request("POST", f"/slots/{id_slot}?action=erase", {})
