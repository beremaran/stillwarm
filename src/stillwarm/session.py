"""The stillwarm session wrapper — the core loop.

restore-before-request, save-after-response, against a running llama-server:

    server  = LlamaServer("http://127.0.0.1:8080")
    profile = ServerProfile(model_sha256=..., flash_attn="on", cache_type_k="f16",
                            cache_type_v="f16", swa_full=False, ctx_size=16384)
    sw = Session(server, profile, cache_dir="~/.stillwarm", server_build_tag="b9871")
    reply = sw.ask("sess-1", document + question, n_predict=256)   # cold: prefill + save
    ... server restarts ...
    reply = sw.ask("sess-1", document + question2, n_predict=256)  # restore + answer

Design decisions (each grounded in a measured stillwarm-bench result):
  * cache key = model SHA + fa + ctk/ctv + swa_full + tokenized-prefix hash
    (Block D: -ngl excluded; ctx via fit rule; build tag advisory).
  * saves happen at the PREFIX state, BEFORE probe generation (save_point=doc_prefill):
    the file must represent the pre-probe state.
  * --verify is a DETERMINISM check against the save-time probe hash — deliberately
    NOT cold-equivalence, which measurably fails for quantized caches (Block C).
  * after every restore, the first request's server-reported prompt_n is inspected;
    near-full re-prefill triggers a loud SILENTLY-INEFFECTIVE warning (Gate-0 finding).
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .compat import ServerProfile, check, check_build, REFUSE
from .server import (LlamaServer, ServerError, PROBE_SAMPLER, PROBE_N_PREDICT,
                     DEFAULT_SAMPLER)
from .sidecar import Sidecar, sidecar_path_for
from . import store as store_mod


class RefusedRestore(RuntimeError):
    """The sidecar says this file must not be loaded into this server."""


class VerifyFailed(RuntimeError):
    """--verify: the restored state did not reproduce the save-time probe."""


class IneffectiveRestoreWarning(UserWarning):
    """Restore 'succeeded' but the next request re-prefilled instead of reusing."""


@dataclass
class AskResult:
    content: str
    timings: dict
    restored: bool          # this request was served from a disk restore
    verified: Optional[bool]  # None = --verify not requested
    prompt_n: Optional[int]
    warning: str = ""


def _hash_tokens(tokens) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


class Session:
    def __init__(self, server: LlamaServer, profile: ServerProfile, cache_dir,
                 *, server_build_tag: str = "", disk_budget_gb: Optional[float] = None,
                 verify: bool = False, id_slot: int = 0):
        self.server = server
        self.profile = profile
        self.cache_dir = Path(os.path.expanduser(str(cache_dir)))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.server_build_tag = server_build_tag
        self.disk_budget_gb = disk_budget_gb
        self.verify = verify
        self.id_slot = id_slot
        self._live_prefix: dict = {}      # session_id -> prefix text currently in the slot
        self._loaded: Optional[str] = None  # session_id currently in the slot

    # -- paths ---------------------------------------------------------------
    def _bin_name(self, session_id: str) -> str:
        return f"{session_id}.bin"

    def _bin_path(self, session_id: str) -> Path:
        return self.cache_dir / self._bin_name(session_id)

    def _prefix_path(self, session_id: str) -> Path:
        return self.cache_dir / f"{session_id}.prefix.txt"

    # -- key material ----------------------------------------------------------
    def _prefix_hash(self, prefix_text: str) -> str:
        toks = self.server.tokenize(prefix_text)
        return _hash_tokens(toks)

    # ==========================================================================
    # THE CORE LOOP
    # ==========================================================================
    def ask(self, session_id: str, full_prompt: str, n_predict: int = 256,
            *, sampler: Optional[dict] = None, verify: Optional[bool] = None) -> AskResult:
        """restore-before-request -> completion -> save-after-response."""
        do_verify = self.verify if verify is None else verify
        restored, verified, warning = False, None, ""

        if self._loaded != session_id and self._bin_path(session_id).exists():
            verified = self.restore(session_id, verify=do_verify)
            restored = True

        res = self.server.completion(full_prompt, n_predict, cache_prompt=True,
                                     id_slot=self.id_slot,
                                     sampler=sampler or dict(DEFAULT_SAMPLER))
        prompt_n = (res.get("timings") or {}).get("prompt_n")

        if restored:
            warning = self._ineffective_guard(session_id, full_prompt, prompt_n)

        # the new prefix = everything the slot now holds (prompt + generated reply)
        new_prefix = full_prompt + res.get("content", "")
        self._live_prefix[session_id] = new_prefix
        self._loaded = session_id
        # the slot state now ENDS WITH the sampled reply -> after_generation state
        generated_n = (res.get("timings") or {}).get("predicted_n") or 0
        self.save(session_id, generated_tokens=generated_n)

        if self.disk_budget_gb is not None:
            store_mod.prune(self.cache_dir, self.disk_budget_gb)

        return AskResult(content=res.get("content", ""), timings=res.get("timings", {}),
                         restored=restored, verified=verified, prompt_n=prompt_n,
                         warning=warning)

    # ==========================================================================
    # SAVE — file is written at the PRE-probe state (save before probe; critical)
    # ==========================================================================
    def save(self, session_id: str, *, generated_tokens: int = 0) -> Sidecar:
        """`generated_tokens` = sampled tokens the slot state currently ends with
        (0 for a pure prefill/prime save). It determines the honest save_point label:
        an after_generation state is NOT a strict prefix of a future prompt that asks
        a different question, which on SWA models forces an impossible rollback
        (measured: stillwarm-bench WS1.4)."""
        prefix = self._live_prefix.get(session_id)
        if prefix is None:
            raise ValueError(f"session {session_id!r} has no live state to save")
        binname = self._bin_name(session_id)
        save_point = "after_generation" if generated_tokens > 0 else "doc_prefill"

        # 1. SAVE FIRST: the file must represent the pre-probe state.
        save = self.server.slot_save(binname, self.id_slot)

        # 2. Probe AFTER the save: 64 greedy tokens continuing the prefix.
        #    (This advances the live KV state; the FILE is already sealed. The next
        #    user request re-trims the probe tail via normal prefix matching.)
        probe = self.server.completion(prefix, PROBE_N_PREDICT, cache_prompt=True,
                                       id_slot=self.id_slot, sampler=dict(PROBE_SAMPLER))
        probe_hash = _hash_tokens(probe.get("tokens") or probe.get("content"))

        # 3. Sidecar + retained prefix text (needed to re-run the probe at restore).
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
        # atomic prefix write (same temp + os.replace pattern; Windows-safe)
        ptmp = self._prefix_path(session_id).with_name(self._prefix_path(session_id).name + ".tmp")
        ptmp.write_text(prefix, encoding="utf-8")
        os.replace(ptmp, self._prefix_path(session_id))
        return sc

    # ==========================================================================
    # RESTORE (+ optional --verify: restore -> probe -> compare -> RE-RESTORE)
    # ==========================================================================
    def restore(self, session_id: str, *, verify: bool = False) -> Optional[bool]:
        binname = self._bin_name(session_id)
        sc = Sidecar.read(sidecar_path_for(self._bin_path(session_id)))

        # refusal contract BEFORE touching the server
        decision = check(sc, self.profile)
        if decision.verdict == REFUSE:
            raise RefusedRestore("; ".join(decision.refusals))
        adv = check_build(sc, self.server_build_tag)
        for a in adv.advisories:
            warnings.warn(a, UserWarning, stacklevel=2)

        self.server.slot_erase(self.id_slot)
        self.server.slot_restore(binname, self.id_slot)
        verified: Optional[bool] = None

        if verify:
            # Cost (measured, 8K/M3 Max): ~2x restore + ~1.5 s of 64-token decode.
            prefix = self._prefix_path(session_id).read_text(encoding="utf-8")
            probe = self.server.completion(
                prefix, sc.probe_n_predict or PROBE_N_PREDICT, cache_prompt=True,
                id_slot=self.id_slot, sampler=json.loads(sc.probe_sampler_json or "null")
                or dict(PROBE_SAMPLER))
            got = _hash_tokens(probe.get("tokens") or probe.get("content"))
            verified = got == sc.probe_hash
            # RE-RESTORE so the served state is exactly the restored state,
            # never the post-probe state.
            self.server.slot_erase(self.id_slot)
            self.server.slot_restore(binname, self.id_slot)
            if not verified:
                raise VerifyFailed(
                    f"restored state failed the determinism probe: expected "
                    f"{sc.probe_hash[:16]}..., got {got[:16]}... — the file, model, or "
                    f"server config does not reproduce the save-time continuation")

        self._live_prefix[session_id] = self._prefix_path(session_id).read_text(encoding="utf-8") \
            if self._prefix_path(session_id).exists() else ""
        self._loaded = session_id
        return verified

    # ==========================================================================
    # SILENTLY-INEFFECTIVE GUARD (Gate-0 finding as a runtime feature)
    # ==========================================================================
    def _ineffective_guard(self, session_id: str, full_prompt: str,
                           prompt_n: Optional[int]) -> str:
        """After a restore, the first request should only prefill the NEW tail.
        If the server reports (near-)full re-prefill, the restore was silently
        ineffective — warn loudly with the measured likely cause."""
        if prompt_n is None:
            return ""
        try:
            full_tokens = len(self.server.tokenize(full_prompt))
        except ServerError:
            return ""
        # reuse was effective if most of the prompt came from the restored cache
        if prompt_n <= max(0.5 * full_tokens, 64):
            return ""
        save_point = ""
        try:
            save_point = Sidecar.read(sidecar_path_for(self._bin_path(session_id))).save_point
        except Exception:
            pass
        causes = ["an SWA model (Gemma family) without --swa-full on the server",
                  "a prompt that does not extend the saved prefix"]
        if save_point == "after_generation":
            causes.append(
                "the saved state ends with generated tokens (save_point=after_generation); "
                "a new question that diverges from them forces a rollback, which SWA caches "
                "without --swa-full cannot do (measured: full re-prefill)")
        msg = (f"restore of session {session_id!r} was SILENTLY INEFFECTIVE: the first "
               f"request re-prefilled {prompt_n}/{full_tokens} tokens instead of reusing "
               f"the restored cache. Measured likely causes: {'; '.join(causes)}. "
               f"You paid full cold-start cost despite the restore.")
        warnings.warn(msg, IneffectiveRestoreWarning, stacklevel=3)
        return msg
