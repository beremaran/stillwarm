# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

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
    TIME. `verify` regenerates the probe after a restore and compares
    against THIS hash — a determinism check, deliberately NOT cold-equivalence
    (which measurably fails for quantized caches without any corruption).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_VERSION = 1
SIDECAR_SUFFIX = ".stillwarm.json"


@dataclasses.dataclass
class Sidecar:
    """Everything needed to decide whether a save file may be restored."""

    model_sha256: str
    flash_attn: str
    cache_type_k: str
    cache_type_v: str
    swa_full: bool
    n_saved_tokens: int
    prompt_prefix_sha256: str

    build_tag: str = ""
    llamacpp_commit: str = ""

    probe_hash: str = ""
    probe_n_predict: int = 0
    probe_sampler_json: str = ""
    save_point: str = "doc_prefill"
    save_file: str = ""
    size_bytes: int = 0
    created_at: str = ""
    sidecar_version: int = SIDECAR_VERSION

    def __post_init__(self) -> None:
        """Validate required fields and normalize values."""
        if self.flash_attn not in ("on", "off"):
            msg = (
                f"flash_attn must be resolved to 'on' or 'off' before saving "
                f"(got {self.flash_attn!r}); 'auto' would make the compat check meaningless"
            )
            raise ValueError(
                msg,
            )
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def to_dict(self) -> dict:
        """Convert sidecar to a JSON-serializable dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Sidecar:
        """Construct a Sidecar from a dict, validating the schema version."""
        version = d.get("sidecar_version")
        if version != SIDECAR_VERSION:
            msg = f"sidecar_version {version!r} not supported (expected {SIDECAR_VERSION})"
            raise IncompatibleSidecarError(
                msg,
            )
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def write(self, path: str | Path) -> Path:
        """Write sidecar as JSON atomically to the given path."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> Sidecar:
        """Read and parse a sidecar JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def cache_key(self) -> str:
        """Compute the cache key from material sidecar fields.

        SHA-256 over exactly the fields the portability matrix showed matter:
        model SHA + fa + ctk/ctv + swa_full + tokenized prefix. -ngl and ctx are
        deliberately excluded (measured irrelevant / handled by the ctx-fit rule).
        """
        material = "\x1f".join(
            [
                self.model_sha256,
                self.flash_attn,
                self.cache_type_k,
                self.cache_type_v,
                "swa_full=1" if self.swa_full else "swa_full=0",
                self.prompt_prefix_sha256,
            ],
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class IncompatibleSidecarError(ValueError):
    """Raised when a sidecar cannot even be parsed under this schema version."""


def sidecar_path_for(save_file: str | Path) -> Path:
    """cache.bin -> cache.bin.stillwarm.json (same directory)."""
    p = Path(save_file)
    return p.with_name(p.name + SIDECAR_SUFFIX)
