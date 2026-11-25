# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""The stillwarm session wrapper — the core loop.

restore-before-request, save-after-response, against a running llama-server:

    server  = LlamaServer("http://127.0.0.1:8080")  # or any reachable http(s) URL, remote included
    profile = ServerProfile(model_sha256=..., flash_attn="on", cache_type_k="f16",
                            cache_type_v="f16", swa_full=False, ctx_size=16384)
    sw = Session(server, profile, cache_dir, server_build_tag="<your build tag>")
    # cache_dir: any directory you choose; it must equal the directory you
    # pass to llama-server's --slot-save-path. The library bakes in no default.
    reply = sw.ask("sess-1", document + question, n_predict=256)
    ... server restarts ...
    reply = sw.ask("sess-1", document + question2, n_predict=256)

Design decisions (each grounded in measured llama.cpp behavior):
  * cache key = model SHA + fa + ctk/ctv + swa_full + tokenized-prefix hash
    (-ngl excluded; ctx via fit rule; build tag advisory).
  * saves happen at the PREFIX state, BEFORE probe generation (save_point=doc_prefill):
    the file must represent the pre-probe state.
  * save_chat (the proxy path) uses the chat probe: it re-POSTs the session's
    own chat request with generation pinned greedy (chat_probe_body), so the
    server re-renders the identical template and the probe continues from
    exactly the saved state — the same teeth as the /completion probe.
  * --verify is a DETERMINISM check against the save-time probe hash — deliberately
    NOT cold-equivalence, which measurably fails for quantized caches.
  * after every restore, the first request's server-reported prompt_n is inspected;
    near-full re-prefill triggers a loud SILENTLY-INEFFECTIVE warning (a measured
    llama.cpp failure mode).
"""

import contextlib
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import orjson

from . import store as store_mod
from .compat import REFUSE, ServerProfile, check, check_build
from .server import DEFAULT_SAMPLER, PROBE_N_PREDICT, PROBE_SAMPLER, LlamaServer, ServerError
from .settings import get_settings
from .sidecar import Sidecar, sidecar_path_for

# Default generation length for ask() when the caller does not choose one.
DEFAULT_N_PREDICT = 256

# Chat probe (save_chat): the probe re-POSTs the session's own chat request
# with generation pinned, so the server re-renders the identical template and
# the probe continues from exactly the saved state. CHAT_PROBE_SAMPLER is
# recorded in the sidecar per save, like PROBE_SAMPLER for /completion probes.
CHAT_PROBE_SAMPLER: dict[str, Any] = {
    "temperature": 0,
    "top_k": 1,
    "ignore_eos": True,
    "max_tokens": PROBE_N_PREDICT,
}
# The probe must not inherit a client-request timeout: a wedged server would
# otherwise hold the proxy's global lock for the whole request timeout. This
# is the probe's dedicated bound (design choice, not a server property).
CHAT_PROBE_TIMEOUT_S = 120.0

# Fields a chat request may carry that affect generation or response shape but
# never the rendered prompt. The probe strips exactly these and pins
# CHAT_PROBE_SAMPLER, so the probe's prompt renders to the identical tokens as
# the original request while its generation is fully deterministic. Field
# names measured against llama.cpp b9871: server-schema.cpp, the
# server_request schema shared by /completion and /v1/chat/completions.
_CHAT_PROBE_STRIP = frozenset(
    {
        # generation limits
        "n_predict",
        "max_tokens",
        "max_completion_tokens",
        "t_max_predict_ms",
        "n_cmpl",
        "n_indent",
        # sampling
        "temperature",
        "top_p",
        "top_k",
        "top_n_sigma",
        "min_p",
        "typical_p",
        "xtc_probability",
        "xtc_threshold",
        "dynatemp_range",
        "dynatemp_exponent",
        "repeat_penalty",
        "repeat_last_n",
        "frequency_penalty",
        "presence_penalty",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "adaptive_target",
        "adaptive_decay",
        "seed",
        "min_keep",
        "n_probs",
        "backend_sampling",
        "post_sampling_probs",
        "ignore_eos",
        "logit_bias",
        "samplers",
        # decoding constraints
        "json_schema",
        "grammar_lazy",
        "grammar_triggers",
        "reasoning_control",
        "reasoning_budget_tokens",
        "reasoning_budget_start_tag",
        "reasoning_budget_end_tag",
        "reasoning_budget_message",
        # streaming
        "stream",
        "stream_options",
        "sse_ping_interval",
    }
)


def chat_probe_body(body: dict[str, Any]) -> dict[str, Any]:
    """Return the pinned, non-streaming probe form of a chat request.

    Keeps every field that can affect the rendered prompt (messages, tools,
    chat_format, stop, ...) verbatim, strips the generation, sampling and
    streaming fields (_CHAT_PROBE_STRIP), and pins CHAT_PROBE_SAMPLER. Feeding
    the result to /v1/chat/completions re-renders the identical prompt tokens
    and generates a deterministic continuation — the chat analogue of the
    /completion probe in save().
    """
    probe = {k: v for k, v in body.items() if k not in _CHAT_PROBE_STRIP}
    probe.update(CHAT_PROBE_SAMPLER)
    probe["stream"] = False
    return probe


class RefusedRestoreError(RuntimeError):
    """The sidecar says this file must not be loaded into this server."""


class VerifyFailedError(RuntimeError):
    """--verify: the restored state did not reproduce the save-time probe."""


class IneffectiveRestoreWarning(UserWarning):
    """Restore 'succeeded' but the next request re-prefilled instead of reusing."""


@dataclass
class AskResult:
    """Result of an ask() call."""

    content: str
    timings: dict[str, Any]
    restored: bool
    verified: bool | None
    prompt_n: int | None
    warning: str = ""


def _hash_tokens(tokens: list[int]) -> str:
    # orjson.dumps matches stdlib compact JSON for int lists.
    return hashlib.sha256(orjson.dumps(tokens)).hexdigest()


class Session:
    """Wrapper around llama-server that auto-restores and saves KV-cache sessions."""

    def __init__(  # noqa: PLR0913 — API signature; all params are meaningful
        self,
        server: LlamaServer,
        profile: ServerProfile,
        cache_dir: str,
        *,
        server_build_tag: str = "",
        disk_budget_gb: float | None = None,
        verify: bool = False,
        id_slot: int = 0,
    ) -> None:
        """Initialize the session wrapper.

        Args:
            server: The LlamaServer HTTP client.
            profile: The server's profile (model, cache types, etc.).
            cache_dir: Directory for storing session KV-cache files.
            server_build_tag: The llama.cpp build tag for advisory warnings.
            disk_budget_gb: Maximum disk usage for all saved sessions.
            verify: Whether to verify restores against probe hashes.
            id_slot: The llama-server slot ID to use.

        The silently-ineffective guard thresholds (measured defaults) can be
        overridden with STILLWARM_GUARD_RATIO / STILLWARM_GUARD_MIN_TOKENS env
        vars, or via a .env file loaded by pydantic-settings. Malformed values
        raise at construction.

        """
        settings = get_settings()
        self.server = server
        self.profile = profile
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.server_build_tag = server_build_tag
        self.disk_budget_gb = disk_budget_gb or settings.disk_budget_gb
        self.verify = verify
        self.id_slot = id_slot
        self._guard_ratio = settings.guard_ratio
        self._guard_min_tokens = settings.guard_min_tokens
        self._live_prefix: dict[str, str] = {}
        self._loaded: str | None = None

    def _bin_name(self, session_id: str) -> str:
        return f"{session_id}.bin"

    def _bin_path(self, session_id: str) -> Path:
        return self.cache_dir / self._bin_name(session_id)

    def _prefix_path(self, session_id: str) -> Path:
        return self.cache_dir / f"{session_id}.prefix.txt"

    def _prefix_hash(self, prefix_text: str) -> str:
        toks = self.server.tokenize(prefix_text)
        return _hash_tokens(toks)

    def ask(
        self,
        session_id: str,
        full_prompt: str,
        n_predict: int = DEFAULT_N_PREDICT,
        *,
        sampler: dict[str, Any] | None = None,
        verify: bool | None = None,
    ) -> AskResult:
        """restore-before-request -> completion -> save-after-response."""
        do_verify = self.verify if verify is None else verify
        restored, verified, warning = False, None, ""

        if self._loaded != session_id and self._bin_path(session_id).exists():
            verified = self.restore(session_id, verify=do_verify)
            restored = True

        res = self.server.completion(
            full_prompt,
            n_predict,
            cache_prompt=True,
            id_slot=self.id_slot,
            sampler=sampler or dict(DEFAULT_SAMPLER),
        )
        prompt_n = (res.get("timings") or {}).get("prompt_n")

        if restored:
            warning = self._ineffective_guard(session_id, full_prompt, prompt_n)

        new_prefix = full_prompt + res.get("content", "")
        self._live_prefix[session_id] = new_prefix
        self._loaded = session_id
        generated_n = (res.get("timings") or {}).get("predicted_n") or 0
        self.save(session_id, generated_tokens=generated_n)

        if self.disk_budget_gb is not None:
            store_mod.prune(str(self.cache_dir), self.disk_budget_gb)

        return AskResult(
            content=res.get("content", ""),
            timings=res.get("timings", {}),
            restored=restored,
            verified=verified,
            prompt_n=prompt_n,
            warning=warning,
        )

    def save(self, session_id: str, *, generated_tokens: int = 0) -> Sidecar:
        """Save the slot's KV cache to disk and write its sidecar.

        `generated_tokens` = sampled tokens the slot state currently ends with
        (0 for a pure prefill/prime save). It determines the honest save_point label:
        an after_generation state is NOT a strict prefix of a future prompt that asks
        a different question, which on SWA models forces an impossible rollback.
        """
        prefix = self._live_prefix.get(session_id)
        if prefix is None:
            msg = f"session {session_id!r} has no live state to save"
            raise ValueError(msg)
        binname = self._bin_name(session_id)
        save_point = "after_generation" if generated_tokens > 0 else "doc_prefill"

        # 1. SAVE FIRST: the file must represent the pre-probe state.
        save = self.server.slot_save(binname, self.id_slot)

        probe = self.server.completion(
            prefix,
            PROBE_N_PREDICT,
            cache_prompt=True,
            id_slot=self.id_slot,
            sampler=dict(PROBE_SAMPLER),
        )
        probe_hash = _hash_tokens(cast("list[int]", probe.get("tokens") or []))

        sc = Sidecar(
            model_sha256=self.profile.model_sha256,
            flash_attn=self.profile.flash_attn,
            cache_type_k=self.profile.cache_type_k,
            cache_type_v=self.profile.cache_type_v,
            swa_full=self.profile.swa_full,
            n_saved_tokens=save["n_saved"],
            prompt_prefix_sha256=self._prefix_hash(prefix),
            build_tag=self.server_build_tag,
            probe_hash=probe_hash,
            probe_n_predict=PROBE_N_PREDICT,
            probe_sampler_json=orjson.dumps(PROBE_SAMPLER).decode(),
            save_point=save_point,
            save_file=binname,
            size_bytes=save["n_written"],
        )
        sc.write(sidecar_path_for(self._bin_path(session_id)))
        ptmp = self._prefix_path(session_id).with_name(self._prefix_path(session_id).name + ".tmp")
        ptmp.write_text(prefix, encoding="utf-8")
        ptmp.replace(self._prefix_path(session_id))
        return sc

    def save_chat(
        self,
        session_id: str,
        probe_body: dict[str, Any] | None,
        *,
        generated_tokens: int = 0,
    ) -> Sidecar:
        """Save the slot's KV cache for a proxied chat session, with the chat probe.

        Mirrors save(): slot_save runs BEFORE the probe, so the file holds the
        pre-probe state. The probe re-POSTs the session's chat request with
        generation pinned greedy (chat_probe_body); the server re-renders the
        identical chat template, the probe continues from exactly the saved
        state, and its message hash keeps verify's teeth. probe_body=None
        records an unprobed sidecar (empty probe fields, empty .prefix.txt);
        restore(verify=True) on such a sidecar returns None — nothing to
        verify.

        On probe failure the half-written triple is deleted (a save without a
        valid probe record is not a restorable session) and the error is
        re-raised for the caller to log; the session reverts to unsaved and
        the next request rebuilds it.
        """
        binname = self._bin_name(session_id)
        save_point = "after_generation" if generated_tokens > 0 else "doc_prefill"

        # 1. SAVE FIRST: the file must represent the pre-probe state.
        save = self.server.slot_save(binname, self.id_slot)

        probe_hash = ""
        probe_sampler_json = ""
        prefix_bytes = b""
        if probe_body is not None:
            probe_input = chat_probe_body(probe_body)
            probe_input.setdefault("id_slot", self.id_slot)
            try:
                probe = self.server.chat(probe_input, timeout=CHAT_PROBE_TIMEOUT_S)
            except (ServerError, httpx.RequestError):
                self._delete_triple(session_id)
                raise
            message = cast("list[Any]", probe.get("choices") or [{}])[0].get("message")
            canonical = orjson.dumps(message, option=orjson.OPT_SORT_KEYS)
            probe_hash = hashlib.sha256(canonical).hexdigest()
            probe_sampler_json = orjson.dumps(CHAT_PROBE_SAMPLER).decode()
            prefix_bytes = orjson.dumps(probe_input, option=orjson.OPT_SORT_KEYS)

        sc = Sidecar(
            model_sha256=self.profile.model_sha256,
            flash_attn=self.profile.flash_attn,
            cache_type_k=self.profile.cache_type_k,
            cache_type_v=self.profile.cache_type_v,
            swa_full=self.profile.swa_full,
            n_saved_tokens=save["n_saved"],
            prompt_prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
            build_tag=self.server_build_tag,
            probe_hash=probe_hash,
            probe_n_predict=PROBE_N_PREDICT if probe_body is not None else 0,
            probe_sampler_json=probe_sampler_json,
            probe_kind="chat",
            save_point=save_point,
            save_file=binname,
            size_bytes=save["n_written"],
        )
        sc.write(sidecar_path_for(self._bin_path(session_id)))
        ppath = self._prefix_path(session_id)
        ptmp = ppath.with_name(ppath.name + ".tmp")
        ptmp.write_bytes(prefix_bytes)
        ptmp.replace(ppath)
        return sc

    def _delete_triple(self, session_id: str) -> None:
        """Remove the whole session triple; a half-save is not a session."""
        for path in (
            self._bin_path(session_id),
            sidecar_path_for(self._bin_path(session_id)),
            self._prefix_path(session_id),
        ):
            with contextlib.suppress(OSError):
                path.unlink()

    def restore(self, session_id: str, *, verify: bool = False) -> bool | None:
        """Restore a saved session into the server slot.

        If verify=True, regenerates the probe and compares against the
        save-time probe hash before re-restoring. The probe is replayed the
        way it was recorded (sidecar probe_kind): "chat" saves re-POST the
        pinned chat request from .prefix.txt via /v1/chat/completions and
        hash choices[0].message; "completion" saves feed the raw prefix text
        to /completion and hash the tokens. An unprobed sidecar (empty
        probe_hash, written by save_chat with probe_body=None) short-circuits:
        no probe is sent and the return value stays None — nothing to verify.
        """
        binname = self._bin_name(session_id)
        sc = Sidecar.read(sidecar_path_for(self._bin_path(session_id)))

        decision = check(sc, self.profile)
        if decision.verdict == REFUSE:
            raise RefusedRestoreError("; ".join(decision.refusals))
        adv = check_build(sc, self.server_build_tag)
        for a in adv.advisories:
            warnings.warn(a, UserWarning, stacklevel=2)

        self.server.slot_erase(self.id_slot)
        self.server.slot_restore(binname, self.id_slot)
        verified: bool | None = None

        if verify and sc.probe_hash:
            prefix_path = self._prefix_path(session_id)
            if sc.probe_kind == "chat":
                # Cost: one pinned chat request against the restored state.
                # Replays .prefix.txt exactly as save_chat wrote it — the pinned
                # sampler fields are already in the body.
                probe_input = cast(
                    "dict[str, Any]",
                    orjson.loads(prefix_path.read_bytes() or b"{}"),
                )
                probe_input.setdefault("id_slot", self.id_slot)
                probe = self.server.chat(probe_input, timeout=CHAT_PROBE_TIMEOUT_S)
                message = cast("list[Any]", probe.get("choices") or [{}])[0].get("message")
                canonical = orjson.dumps(message, option=orjson.OPT_SORT_KEYS)
                got = hashlib.sha256(canonical).hexdigest()
            else:
                # Cost: ~2x restore, plus a 64-token decode (~1.5 s for an 8B model).
                prefix = prefix_path.read_text(encoding="utf-8")
                probe = self.server.completion(
                    prefix,
                    sc.probe_n_predict or PROBE_N_PREDICT,
                    cache_prompt=True,
                    id_slot=self.id_slot,
                    sampler=orjson.loads(sc.probe_sampler_json or "null") or dict(PROBE_SAMPLER),
                )
                got = _hash_tokens(cast("list[int]", probe.get("tokens") or []))
            verified = got == sc.probe_hash
            self.server.slot_erase(self.id_slot)
            self.server.slot_restore(binname, self.id_slot)
            if not verified:
                msg = (
                    f"restored state failed the determinism probe: expected "
                    f"{sc.probe_hash[:16]}..., got {got[:16]}... — the file, model, or "
                    f"server config does not reproduce the save-time continuation"
                )
                raise VerifyFailedError(
                    msg,
                )

        self._live_prefix[session_id] = (
            self._prefix_path(session_id).read_text(encoding="utf-8")
            if self._prefix_path(session_id).exists()
            else ""
        )
        self._loaded = session_id
        return verified

    def _ineffective_guard(self, session_id: str, full_prompt: str, prompt_n: int | None) -> str:
        """Detect and warn when a restore was silently ineffective.

        After a restore, the first request should only prefill the NEW tail.
        If the server reports (near-)full re-prefill, the restore was silently
        ineffective — warn loudly with the measured likely cause.
        """
        if prompt_n is None:
            return ""
        try:
            full_tokens = len(self.server.tokenize(full_prompt))
        except ServerError:
            return ""
        if prompt_n <= max(self._guard_ratio * full_tokens, self._guard_min_tokens):
            return ""
        save_point = ""
        with contextlib.suppress(Exception):
            save_point = Sidecar.read(sidecar_path_for(self._bin_path(session_id))).save_point
        causes = [
            "an SWA model (Gemma family) without --swa-full on the server",
            "a prompt that does not extend the saved prefix",
        ]
        if save_point == "after_generation":
            causes.append(
                "the saved state ends with generated tokens (save_point=after_generation); "
                "a new question that diverges from them forces a rollback, which SWA caches "
                "without --swa-full cannot do (measured: full re-prefill)",
            )
        msg = (
            f"restore of session {session_id!r} was SILENTLY INEFFECTIVE: the first "
            f"request re-prefilled {prompt_n}/{full_tokens} tokens instead of reusing "
            f"the restored cache. Measured likely causes: {'; '.join(causes)}. "
            f"You paid full cold-start cost despite the restore."
        )
        warnings.warn(msg, IneffectiveRestoreWarning, stacklevel=3)
        return msg
