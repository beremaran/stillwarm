# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""stillwarm proxy — stillwarm's loop applied to /v1/chat/completions.

An HTTP middleware between any OpenAI-compatible client and ONE single-slot
llama-server (--parallel 1). Per chat request, under one global lock:
restore the session's saved KV state (or start cold), forward the request
verbatim, then save the slot back — the proxy never mutates the client's
request or the server's response bytes. Every other path passes through
unchanged.

Session identity is the X-Stillwarm-Session header when present (charset
[A-Za-z0-9._-], max 64, else 400 — the id becomes a filename), otherwise the
SHA-256 of the canonical JSON of [system message, first user message]; the
derived id is recomputed from the resent history each request, so a client
that compacts or edits its history starts a new session (the old file is
pruned by LRU, never misloaded).

The response carries one stillwarm header, X-Stillwarm-Restore:
  cold    — no saved file; forwarded as-is
  refused — the sidecar refused this server; erased and forwarded cold
  ok      — the saved state was restored

Measured against llama.cpp b9871 (the compatibility bar for every claim):
  * /v1/chat/completions honors id_slot from the raw request JSON
    (server-context.cpp, shared prompt-processing path) — the probe can pin
    the slot explicitly.
  * usage = {completion_tokens, prompt_tokens, total_tokens,
    prompt_tokens_details: {cached_tokens}}; non-stream always, stream only
    in the final chunk when stream_options.include_usage is set
    (server-task.cpp). cached_tokens == n_past (server-context.cpp) — the
    input of the ineffective-restore guard.
  * timings (prompt_n, predicted_n, ...) is attached to the final stream
    chunk (the usage chunk when include_usage) and to the non-stream body.
  * SSE frames are `data: <json>` lines terminated by a blank line; the
    stream ends with `data: [DONE]`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import hmac
import logging
import re
import time
from typing import TYPE_CHECKING, Any, cast

import anyio
import httpx
import orjson
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import store
from .compat import ServerProfile
from .server import LlamaServer, ServerError
from .session import RefusedRestoreError, Session
from .settings import get_settings
from .sidecar import Sidecar, sidecar_path_for

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from starlette.datastructures import Headers
    from starlette.requests import Request

logger = logging.getLogger("stillwarm.proxy")

# Session ids become filenames (<id>.bin, <id>.bin.stillwarm.json,
# <id>.prefix.txt); the charset is the path-traversal guard.
SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
# POST /slots/{id}?action=save|restore|erase touches the slot cache, so it
# serializes with the chat path; every other path passes through unlocked.
_SLOTS_CACHE_OP_RE = re.compile(r"/slots/\d+\Z")
_SLOTS_CACHE_ACTIONS = frozenset({"save", "restore", "erase"})
_DROP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "content-encoding",  # the body is buffered and re-sent decoded
        "connection",
        "keep-alive",
        "proxy-authorization",
        "te",
        "upgrade",
    }
)
_DROP_RESPONSE_HEADERS = frozenset(
    {
        b"content-length",  # re-chunked on the way out; the body bytes are unchanged
        b"transfer-encoding",
        b"connection",
        b"keep-alive",
        b"te",
        b"trailer",
        b"upgrade",
    }
)
# SSE frame separator = a blank line; the earliest occurrence wins.
_LF_FRAME_SEP = b"\n\n"
_CRLF_FRAME_SEP = b"\r\n\r\n"
# The proxy's own credential, enforced here and never forwarded upstream.
_BEARER_RE = re.compile(r"bearer\s+(.+)", re.IGNORECASE)
_STATUS_OK = 200
# Starlette defaults a function endpoint to GET-only, so the catch-all must
# name every method it forwards (HEAD is added by Starlette for GET).
_PASSTHROUGH_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"})


class ProxyStartupError(RuntimeError):
    """The proxy cannot start: bad config, or the upstream is not usable."""


@dataclasses.dataclass(frozen=True)
class ProxyConfig:
    """Resolved proxy configuration (CLI flags over STILLWARM_* env vars)."""

    base_url: str
    cache_dir: str
    model_sha256: str
    flash_attn: str
    cache_type_k: str
    cache_type_v: str
    swa_full: bool
    ctx_size: int  # 0 = take it from GET /props (n_ctx)
    host: str
    port: int
    token: str
    timeout: float
    disk_budget_gb: float | None
    no_probe: bool


class ProxySettings(BaseSettings):
    """STILLWARM_-prefixed defaults for `stillwarm proxy`.

    CLI flags override these; fields left empty here are a startup error
    (the profile cannot be guessed: a sidecar records exactly these values,
    and the REFUSE contract compares against them).
    """

    model_config = SettingsConfigDict(
        env_prefix="STILLWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "http://127.0.0.1:8080"
    cache_dir: str = ""
    model_sha256: str = ""
    flash_attn: str = ""
    cache_type_k: str = ""
    cache_type_v: str = ""
    swa_full: bool = False
    ctx_size: int = 0
    host: str = "127.0.0.1"
    port: int = 8081
    token: str = ""
    # Upstream chat requests may run for a long time; this is the proxy's
    # default, not the library's (LlamaServer.DEFAULT_TIMEOUT_S).
    timeout: float = 3600.0
    disk_budget_gb: float | None = None
    no_probe: bool = False


def _role(message: object) -> str:
    if isinstance(message, dict):
        role = message.get("role")
        if isinstance(role, str):
            return role
    return ""


def derived_session_id(body: dict[str, Any]) -> str:
    """Return the derived session id for a chat request body.

    SHA-256 of the canonical JSON (orjson, sorted keys) of the [system
    message, first user message] objects (null when absent). Stable across
    turns because clients resend their history; changes when the system
    prompt or the first user message changes, which is the point: a
    different conversation prefix is a different session.
    """
    messages = cast("list[Any]", body.get("messages") or [])
    system = next((m for m in messages if _role(m) == "system"), None)
    user = next((m for m in messages if _role(m) == "user"), None)
    canonical = orjson.dumps([system, user], option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


def _is_sse(content_type: str) -> bool:
    return content_type.lower().startswith("text/event-stream")


class ChatTee:
    """Relay a chat completion response byte-for-byte while extracting state.

    SSE responses are parsed frame by frame (frames end with a blank line;
    each `data:` line is one JSON chunk); the raw frame bytes are relayed
    unchanged. Non-streaming responses are relayed as they arrive and parsed
    once as a whole in finish(). Extraction is defensive: a malformed chunk
    is skipped, never fatal — the client's response comes first.
    """

    def __init__(self, *, sse: bool) -> None:
        """Initialize; sse selects frame parsing over whole-body JSON parsing."""
        self._sse = sse
        self._buf = b""
        self.saw_generation = False
        self.usage: dict[str, Any] = {}
        self.timings: dict[str, Any] = {}
        self._body = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        """Absorb a raw chunk; return the complete frames to relay."""
        if not self._sse:
            self._body += chunk
            return [chunk]
        self._buf += chunk
        frames: list[bytes] = []
        while True:
            sep, idx = self._find_separator()
            if idx < 0:
                return frames
            frame = self._buf[:idx]
            self._parse_frame(frame)
            frames.append(frame + sep)
            self._buf = self._buf[idx + len(sep) :]

    def finish(self) -> list[bytes]:
        """End of stream; return any trailing bytes to relay."""
        if self._sse:
            tail = self._buf
            self._parse_frame(tail)
            self._buf = b""
            return [tail] if tail else []
        self._parse_json(self._body)
        return []

    async def relay(self, resp: httpx.Response) -> AsyncIterator[bytes]:
        """Yield every relay frame of the response while parsing it."""
        async for chunk in resp.aiter_raw():
            for frame in self.feed(chunk):
                yield frame
        for frame in self.finish():
            yield frame

    def _find_separator(self) -> tuple[bytes, int]:
        """Return the earliest frame separator in the buffer: (separator, index)."""
        lf = self._buf.find(_LF_FRAME_SEP)
        crlf = self._buf.find(_CRLF_FRAME_SEP)
        if lf < 0:
            return (_CRLF_FRAME_SEP, crlf)
        if crlf < 0 or lf < crlf:
            return (_LF_FRAME_SEP, lf)
        return (_CRLF_FRAME_SEP, crlf)

    def _parse_frame(self, frame: bytes) -> None:
        for raw in frame.split(b"\n"):
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].lstrip()
            if data == b"[DONE]":
                continue
            with contextlib.suppress(ValueError):
                self._update_chunk(orjson.loads(data))

    def _parse_json(self, body: bytes) -> None:
        with contextlib.suppress(ValueError):
            self._update_chunk(orjson.loads(body))

    def _update_chunk(self, obj: object) -> None:
        if not isinstance(obj, dict):
            return
        usage = obj.get("usage")
        if isinstance(usage, dict):
            self.usage.update(usage)
        timings = obj.get("timings")
        if isinstance(timings, dict):
            self.timings.update(timings)
        choices = obj.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for key in ("delta", "message"):
                payload = choice.get(key)
                if isinstance(payload, dict) and (
                    payload.get("content") or payload.get("tool_calls")
                ):
                    self.saw_generation = True

    def has_save_point(self) -> bool:
        """Return True when the response reached a state worth saving.

        The signal is the server's own count where available
        (usage.completion_tokens), else any content or tool-call delta:
        the slot state then ends with generated tokens.
        """
        return self.generated_tokens() > 0

    def generated_tokens(self) -> int:
        """Completion token count, or a 1/0 sentinel when the response has no usage.

        Streaming without stream_options.include_usage carries no usage, so the
        count falls back to whether any content or tool-call delta was seen;
        1 is a valid sentinel (it only labels the save_point, never a count
        stored in the sidecar).
        """
        count = self.usage.get("completion_tokens")
        if isinstance(count, int) and count > 0:
            return count
        return 1 if self.saw_generation else 0

    def cached_tokens(self) -> int | None:
        """Server-reported prompt tokens served from existing KV, or None.

        usage.prompt_tokens_details.cached_tokens; None when the response
        carries no usage (the ineffective-restore guard then has nothing to
        measure and is skipped).
        """
        details = self.usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            return None
        cached = details.get("cached_tokens")
        return cached if isinstance(cached, int) else None


class ProxyApp:
    """The Starlette middleware: one lock, one upstream, one save directory."""

    def __init__(
        self,
        cfg: ProxyConfig,
        *,
        server: LlamaServer,
        session: Session,
        upstream: httpx.AsyncClient,
    ) -> None:
        """Initialize; the guard thresholds come from the library settings."""
        self.cfg = cfg
        self._base_url = cfg.base_url.rstrip("/")
        self.server = server
        self.session = session
        self.upstream = upstream
        self.lock = asyncio.Lock()
        settings = get_settings()
        self._guard_ratio = settings.guard_ratio
        self._guard_min_tokens = settings.guard_min_tokens

    # -- request handling ---------------------------------------------------

    async def chat(self, request: Request) -> Response:
        """POST /v1/chat/completions: restore-before, forward, save-after."""
        if not self._authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body_bytes = await request.body()
        body = self._parse_chat_body(body_bytes)
        if body is None:
            # Not a chat request the proxy can reason about: pass it through.
            return await self.passthrough(request)
        resolved = self._resolve_session(request, body)
        if isinstance(resolved, JSONResponse):
            return resolved
        session_id = resolved

        await self.lock.acquire()
        try:
            state, sidecar = await anyio.to_thread.run_sync(self._restore_phase, session_id)
        except (ServerError, httpx.RequestError) as e:
            self.lock.release()
            logger.exception("slot restore failed for session %s", session_id)
            return JSONResponse({"error": f"upstream slot error: {e}"}, status_code=502)

        target = self._target_url("/v1/chat/completions")
        try:
            resp = await self._open_upstream("POST", target, body_bytes, request.headers)
        except httpx.RequestError as e:
            self.lock.release()
            logger.exception("upstream unreachable")
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)

        tee = ChatTee(sse=_is_sse(resp.headers.get("content-type", "")))
        headers = self._response_headers(resp.headers)
        headers.append("X-Stillwarm-Restore", state)

        async def generate() -> AsyncIterator[bytes]:
            save_s: float | None = None
            completed = False
            try:
                async for frame in tee.relay(resp):
                    yield frame
                completed = True
                if resp.status_code == _STATUS_OK and tee.has_save_point():
                    generated = tee.generated_tokens()
                    _, save_s = await anyio.to_thread.run_sync(
                        self._save_phase, session_id, body, generated
                    )
                    self._guard(session_id, state, sidecar, tee)
            finally:
                with contextlib.suppress(Exception):
                    await resp.aclose()
                self._log_request(session_id, state, tee, save_s, completed=completed)
                self.lock.release()

        return StreamingResponse(generate(), status_code=resp.status_code, headers=headers)

    async def passthrough(self, request: Request) -> Response:
        """Everything else: forward verbatim; slot cache ops take the lock."""
        if not self._authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.body()
        locked = self._is_slots_cache_op(request)
        if locked:
            await self.lock.acquire()
        url = request.url
        target = self._target_url(url.path, url.query)
        try:
            resp = await self._open_upstream(request.method, target, body, request.headers)
        except httpx.RequestError as e:
            if locked:
                self.lock.release()
            logger.exception("upstream unreachable")
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)

        async def generate() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                with contextlib.suppress(Exception):
                    await resp.aclose()
                if locked:
                    self.lock.release()

        return StreamingResponse(
            generate(), status_code=resp.status_code, headers=self._response_headers(resp.headers)
        )

    # -- phases -------------------------------------------------------------

    def _restore_phase(self, session_id: str) -> tuple[str, Sidecar | None]:
        """Cold / refused / ok, plus the sidecar (for the guard) when ok.

        Raises ServerError / httpx.RequestError when the slot endpoints fail:
        the caller answers 502, no save is attempted.
        """
        bin_path = self._bin_path(session_id)
        if not bin_path.exists():
            return "cold", None
        try:
            sidecar = Sidecar.read(sidecar_path_for(bin_path))
        except (OSError, ValueError) as e:
            # A .bin without a readable sidecar is not a session (the triple
            # is one unit); remove the orphan and start cold.
            logger.warning(
                "session %s has an unreadable sidecar (%s); removing orphan files, starting cold",
                session_id,
                e,
            )
            self._delete_triple(session_id)
            self.server.slot_erase(self.session.id_slot)
            return "cold", None
        try:
            self.session.restore(session_id)
        except RefusedRestoreError as e:
            logger.warning(
                "restore of session %s refused (%s); erasing slot and starting cold",
                session_id,
                e,
            )
            self.server.slot_erase(self.session.id_slot)
            return "refused", None
        return "ok", sidecar

    def _save_phase(
        self, session_id: str, body: dict[str, Any], generated_tokens: int
    ) -> tuple[Sidecar | None, float]:
        """slot_save -> chat probe -> triple -> prune; never raises.

        On any failure the session reverts to unsaved: the next request
        rebuilds it (usually a full prefix reuse from warm VRAM). The client
        already has its response; a save failure must not break the loop.
        """
        started = time.perf_counter()
        probe_body = None if self.cfg.no_probe else body
        try:
            sidecar = self.session.save_chat(
                session_id, probe_body, generated_tokens=generated_tokens
            )
        except (ServerError, httpx.RequestError, OSError, ValueError) as e:
            logger.warning(
                "save failed for session %s: %s; session reverts to unsaved", session_id, e
            )
            return None, time.perf_counter() - started
        if self.cfg.disk_budget_gb is not None:
            try:
                store.prune(self.session.cache_dir, self.cfg.disk_budget_gb)
            except OSError as e:
                logger.warning("prune after save failed: %s", e)
        return sidecar, time.perf_counter() - started

    def _guard(self, session_id: str, state: str, sidecar: Sidecar | None, tee: ChatTee) -> None:
        """Loudly warn when a restored cache was not actually reused.

        The server reports how many prompt tokens it served from existing KV
        (cached_tokens == n_past, b9871). After a successful restore of a
        file holding n_saved_tokens, the server should have matched (most of)
        those; a large shortfall means the restore silently did nothing. The
        thresholds are the library's guard_ratio / guard_min_tokens. When the
        response carries no usage (streaming without include_usage) there is
        nothing to measure, and the check is skipped.
        """
        if state != "ok" or sidecar is None:
            return
        cached = tee.cached_tokens()
        if cached is None:
            return
        n_saved = sidecar.n_saved_tokens
        missing = n_saved - cached
        if missing > max(self._guard_ratio * n_saved, self._guard_min_tokens):
            logger.warning(
                "restore of session %s was SILENTLY INEFFECTIVE: the server reused only %d of "
                "%d saved tokens. Measured likely causes: the request does not extend the saved "
                "prefix (client compaction or history edit); an SWA model without --swa-full; "
                "the saved state ends with generated tokens and the new turn diverges from them",
                session_id,
                cached,
                n_saved,
            )

    def _log_request(
        self,
        session_id: str,
        state: str,
        tee: ChatTee,
        save_s: float | None,
        *,
        completed: bool,
    ) -> None:
        prefill = tee.timings.get("prompt_n")
        cached = tee.cached_tokens()
        generated = tee.generated_tokens()
        logger.info(
            "session=%s restore=%s generated=%s prefill=%s cached=%s save=%s%s",
            session_id[:16],
            state,
            generated or "-",
            prefill if prefill is not None else "-",
            cached if cached is not None else "-",
            f"{save_s:.2f}s" if save_s is not None else "-",
            "" if completed else " (request aborted)",
        )

    # -- plumbing -----------------------------------------------------------

    def _authorized(self, request: Request) -> bool:
        if not self.cfg.token:
            return True
        match = _BEARER_RE.fullmatch(request.headers.get("authorization", ""))
        if match is None:
            return False
        return hmac.compare_digest(match.group(1).encode("utf-8"), self.cfg.token.encode("utf-8"))

    @staticmethod
    def _resolve_session(request: Request, body: dict[str, Any]) -> str | JSONResponse:
        """Return the session id: the validated header, or the derived one; 400 if bad."""
        header_id = request.headers.get("x-stillwarm-session")
        if header_id is None:
            return derived_session_id(body)
        if SESSION_ID_RE.fullmatch(header_id) is None:
            return JSONResponse(
                {"error": "X-Stillwarm-Session must match [A-Za-z0-9._-]{1,64}"},
                status_code=400,
            )
        return header_id

    @staticmethod
    def _parse_chat_body(body_bytes: bytes) -> dict[str, Any] | None:
        """Return the body as a chat request (dict with a messages list), else None."""
        try:
            body = orjson.loads(body_bytes)
        except ValueError:
            return None
        if isinstance(body, dict) and isinstance(body.get("messages"), list):
            return body
        return None

    def _target_url(self, path: str, query: str = "") -> httpx.URL:
        """Return the upstream URL for a client path+query against cfg.base_url."""
        suffix = f"{path}?{query}" if query else path
        return httpx.URL(self._base_url + suffix)

    async def _open_upstream(
        self, method: str, target: httpx.URL, body: bytes, headers: Headers
    ) -> httpx.Response:
        req = self.upstream.build_request(
            method, target, content=body, headers=self._forward_headers(headers)
        )
        return await self.upstream.send(req, stream=True)

    @staticmethod
    def _forward_headers(headers: Headers) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for key, value in headers.items():
            lower = key.lower()
            if lower in _DROP_REQUEST_HEADERS:
                continue
            if lower == "authorization":
                continue  # the client's credential is for the proxy, never forwarded
            out.append((key, value))
        return out

    @staticmethod
    def _response_headers(headers: httpx.Headers) -> MutableHeaders:
        """Return the upstream response headers minus the hop-by-hop ones."""
        raw = [
            (key, value) for key, value in headers.raw if key.lower() not in _DROP_RESPONSE_HEADERS
        ]
        return MutableHeaders(raw=raw)

    def _is_slots_cache_op(self, request: Request) -> bool:
        if request.method != "POST":
            return False
        if not _SLOTS_CACHE_OP_RE.fullmatch(request.url.path):
            return False
        return request.query_params.get("action") in _SLOTS_CACHE_ACTIONS

    def _bin_path(self, session_id: str) -> Path:
        return self.session.cache_dir / f"{session_id}.bin"

    def _delete_triple(self, session_id: str) -> None:
        bin_path = self._bin_path(session_id)
        for path in (
            bin_path,
            sidecar_path_for(bin_path),
            self.session.cache_dir / f"{session_id}.prefix.txt",
        ):
            with contextlib.suppress(OSError):
                path.unlink()


def build_app(
    cfg: ProxyConfig,
    *,
    server: LlamaServer | None = None,
    upstream: httpx.AsyncClient | None = None,
) -> Starlette:
    """Build the proxy app; verify the upstream first.

    GET /props is fetched at startup: unreachable is a hard error, and
    build_info becomes the sidecar build tag while n_ctx (when
    cfg.ctx_size is 0) becomes the profile's ctx size. total_slots != 1
    only warns: the proxy assumes one slot and clobbers it by design.
    """
    if cfg.flash_attn not in ("on", "off"):
        msg = (
            f"flash_attn must be resolved to 'on' or 'off' (got {cfg.flash_attn!r}); "
            "'auto' would make the compat check meaningless"
        )
        raise ProxyStartupError(msg)
    server = server if server is not None else LlamaServer(cfg.base_url, timeout=cfg.timeout)
    props = server.props()
    total_slots = props.get("total_slots")
    if isinstance(total_slots, int) and total_slots != 1:
        logger.warning(
            "upstream reports %d slots; the proxy assumes a single-slot server "
            "(--parallel 1) and saves/restores slot 0 only",
            total_slots,
        )
    ctx_size = cfg.ctx_size
    if ctx_size <= 0:
        n_ctx = props.get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            ctx_size = n_ctx
    if ctx_size <= 0:
        msg = (
            "cannot determine the slot ctx size: pass --ctx-size (or STILLWARM_CTX_SIZE); "
            "the upstream /props response had no usable n_ctx"
        )
        raise ProxyStartupError(msg)
    profile = ServerProfile(
        model_sha256=cfg.model_sha256,
        flash_attn=cfg.flash_attn,
        cache_type_k=cfg.cache_type_k,
        cache_type_v=cfg.cache_type_v,
        swa_full=cfg.swa_full,
        ctx_size=ctx_size,
    )
    session = Session(
        server,
        profile,
        cfg.cache_dir,
        server_build_tag=str(props.get("build_info") or ""),
        disk_budget_gb=cfg.disk_budget_gb,
    )
    upstream = (
        upstream
        if upstream is not None
        else httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.timeout)
    )
    proxy = ProxyApp(cfg, server=server, session=session, upstream=upstream)
    app = Starlette(
        routes=[
            Route("/v1/chat/completions", proxy.chat, methods=["POST"]),
            Route("/{path:path}", proxy.passthrough, methods=_PASSTHROUGH_METHODS),
        ]
    )
    app.state.proxy = proxy
    return app
