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
  * --verify is a DETERMINISM check against the save-time probe hash — deliberately
    NOT cold-equivalence, which measurably fails for quantized caches.
  * after every restore, the first request's server-reported prompt_n is inspected;
    near-full re-prefill triggers a loud SILENTLY-INEFFECTIVE warning (a measured
    llama.cpp failure mode).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import store as store_mod
from .compat import REFUSE, ServerProfile, check, check_build
from .server import DEFAULT_SAMPLER, PROBE_N_PREDICT, PROBE_SAMPLER, LlamaServer, ServerError
from .sidecar import Sidecar, sidecar_path_for

# Default generation length for ask() when the caller does not choose one.
DEFAULT_N_PREDICT = 256

# Silently-ineffective-restore guard: if the first request after a restore
# re-prefills more than this fraction of the prompt — or more than this many
# tokens, whichever is larger — the restore did not take. Override with the
# STILLWARM_GUARD_* env vars.
_INEFFECTIVE_RATIO = 0.5
_INEFFECTIVE_MIN_TOKENS = 64
_ENV_GUARD_RATIO = "STILLWARM_GUARD_RATIO"
_ENV_GUARD_MIN_TOKENS = "STILLWARM_GUARD_MIN_TOKENS"


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
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _env_guard_ratio() -> float:
    """Guard ratio override from the environment, else the measured default."""
    raw = os.environ.get(_ENV_GUARD_RATIO)
    if raw is None:
        return _INEFFECTIVE_RATIO
    try:
        return float(raw)
    except ValueError:
        msg = f"{_ENV_GUARD_RATIO}={raw!r} is not a valid float"
        raise ValueError(msg) from None


def _env_guard_min_tokens() -> int:
    """Guard token floor from the environment, else the measured default."""
    raw = os.environ.get(_ENV_GUARD_MIN_TOKENS)
    if raw is None:
        return _INEFFECTIVE_MIN_TOKENS
    try:
        return int(raw)
    except ValueError:
        msg = f"{_ENV_GUARD_MIN_TOKENS}={raw!r} is not a valid integer"
        raise ValueError(msg) from None


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
        overridden with the STILLWARM_GUARD_RATIO / STILLWARM_GUARD_MIN_TOKENS
        environment variables; malformed values raise at construction.

        """
        self.server = server
        self.profile = profile
        self.cache_dir = Path(str(cache_dir)).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.server_build_tag = server_build_tag
        self.disk_budget_gb = disk_budget_gb
        self.verify = verify
        self.id_slot = id_slot
        self._guard_ratio = _env_guard_ratio()
        self._guard_min_tokens = _env_guard_min_tokens()
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
            probe_sampler_json=json.dumps(PROBE_SAMPLER, sort_keys=True),
            save_point=save_point,
            save_file=binname,
            size_bytes=save["n_written"],
        )
        sc.write(sidecar_path_for(self._bin_path(session_id)))
        ptmp = self._prefix_path(session_id).with_name(self._prefix_path(session_id).name + ".tmp")
        ptmp.write_text(prefix, encoding="utf-8")
        ptmp.replace(self._prefix_path(session_id))
        return sc

    def restore(self, session_id: str, *, verify: bool = False) -> bool | None:
        """Restore a saved session into the server slot.

        If verify=True, regenerates the probe and compares against the
        save-time probe hash before re-restoring.
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

        if verify:
            # Cost: ~2x restore, plus a 64-token decode (~1.5 s for an 8B model).
            prefix = self._prefix_path(session_id).read_text(encoding="utf-8")
            probe = self.server.completion(
                prefix,
                sc.probe_n_predict or PROBE_N_PREDICT,
                cache_prompt=True,
                id_slot=self.id_slot,
                sampler=json.loads(sc.probe_sampler_json or "null") or dict(PROBE_SAMPLER),
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
