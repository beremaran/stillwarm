"""Metadata sidecar — one small JSON file next to every KV-cache save file.

The sidecar is what makes restores SAFE: before loading a save file into
llama-server, stillwarm checks the sidecar against the running server and
REFUSES incompatible files instead of letting them misload silently.

Schema v1 field provenance (every compatibility field is grounded in a measured
stillwarm-bench Block-D outcome, 2026-07-05):
  - flash_attn, cache_type_k/v, swa_full, model_sha256: restore across a
    mismatch either fails or is meaningless -> part of the REFUSE contract.
  - n_saved_tokens: llama-server refuses states larger than the server ctx
    ("ctx-fit rule") -> checked up-front for a clearer error.
  - build_tag / llamacpp_commit: measured STABLE across >=5 weeks of releases
    (b9386<->b9871 both directions) -> ADVISORY warning only, never a refusal.
  - probe_hash: SHA-256 of a 64-token greedy continuation captured AT SAVE
    TIME. `--verify` (Phase 3 later work) regenerates after restore and compares
    against THIS hash — a determinism check, deliberately NOT cold-equivalence
    (which measurably fails for quantized caches without any corruption).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

SIDECAR_VERSION = 1
SIDECAR_SUFFIX = ".stillwarm.json"


@dataclasses.dataclass
class Sidecar:
    """Everything needed to decide whether a save file may be restored."""

    # --- REFUSE-contract fields (mismatch => never load) ---
    model_sha256: str
    flash_attn: str              # "on" | "off"  ("auto" is never written; resolve first)
    cache_type_k: str            # e.g. "f16", "q8_0", "q4_0"
    cache_type_v: str
    swa_full: bool
    n_saved_tokens: int          # ctx-fit rule: refuse if > server ctx
    prompt_prefix_sha256: str    # SHA-256 over the tokenized conversation prefix (cache key part)

    # --- ADVISORY fields (mismatch => warn, don't refuse) ---
    build_tag: str = ""          # e.g. "b9871"
    llamacpp_commit: str = ""

    # --- verification + bookkeeping ---
    probe_hash: str = ""         # save-time greedy continuation hash (empty = no probe)
    probe_n_predict: int = 0     # tokens generated for the probe (0 = no probe)
    probe_sampler_json: str = "" # exact sampler params used for the probe, JSON
    save_point: str = "doc_prefill"   # doc_prefill | after_generation (stillwarm always
                                      # saves BEFORE probing, so state = the prefix)
    save_file: str = ""          # basename of the .bin this sidecar describes
    size_bytes: int = 0
    created_at: str = ""
    sidecar_version: int = SIDECAR_VERSION

    def __post_init__(self) -> None:
        if self.flash_attn not in ("on", "off"):
            raise ValueError(
                f"flash_attn must be resolved to 'on' or 'off' before saving "
                f"(got {self.flash_attn!r}); 'auto' would make the compat check meaningless")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sidecar":
        version = d.get("sidecar_version")
        if version != SIDECAR_VERSION:
            raise IncompatibleSidecar(
                f"sidecar_version {version!r} not supported (expected {SIDECAR_VERSION})")
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def write(self, path: Union[str, Path]) -> Path:
        # atomic on POSIX and Windows: write a sibling temp file, then os.replace
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def read(cls, path: Union[str, Path]) -> "Sidecar":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- cache key ----------------------------------------------------------
    def cache_key(self) -> str:
        """SHA-256 over exactly the fields Block D showed matter (PLAN §7.2):
        model SHA + fa + ctk/ctv + swa_full + tokenized prefix. -ngl and ctx are
        deliberately excluded (measured irrelevant / handled by the ctx-fit rule)."""
        material = "\x1f".join([
            self.model_sha256, self.flash_attn, self.cache_type_k, self.cache_type_v,
            "swa_full=1" if self.swa_full else "swa_full=0", self.prompt_prefix_sha256,
        ])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class IncompatibleSidecar(ValueError):
    """Raised when a sidecar cannot even be parsed under this schema version."""


def sidecar_path_for(save_file: Union[str, Path]) -> Path:
    """cache.bin -> cache.bin.stillwarm.json (same directory)."""
    p = Path(save_file)
    return p.with_name(p.name + SIDECAR_SUFFIX)
