# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""Compatibility decision: may this save file be restored into this server.

Verdicts reflect measured llama.cpp behavior:
  REFUSE    — a mismatch that measurably fails or silently misbehaves; never load.
  ADVISORY  — measurably compatible, but worth a warning (e.g. build drift).
  OK        — no known incompatibility.

The decision is pure and side-effect free; callers act on it.
"""

import dataclasses

from .sidecar import Sidecar

REFUSE = "REFUSE"
ADVISORY = "ADVISORY"
OK = "OK"


@dataclasses.dataclass
class ServerProfile:
    """The relevant properties of the running llama-server."""

    model_sha256: str
    flash_attn: str
    cache_type_k: str
    cache_type_v: str
    swa_full: bool
    ctx_size: int


@dataclasses.dataclass
class Decision:
    """Result of a compatibility check."""

    verdict: str
    refusals: list[str]
    advisories: list[str]

    def __bool__(self) -> bool:
        """Return True if the save file is loadable (OK or ADVISORY)."""
        return self.verdict != REFUSE


def check(sidecar: Sidecar, server: ServerProfile) -> Decision:
    """Check if a saved sidecar is compatible with the running server."""
    refusals: list[str] = []
    advisories: list[str] = []

    if sidecar.model_sha256 != server.model_sha256:
        refusals.append("model mismatch: save was made with a different model file")
    if sidecar.flash_attn != server.flash_attn:
        refusals.append(
            f"flash-attention mismatch (file: {sidecar.flash_attn}, server: "
            f"{server.flash_attn}): measured FAILS-CLEAN in both directions",
        )
    if sidecar.cache_type_k != server.cache_type_k or sidecar.cache_type_v != server.cache_type_v:
        refusals.append(
            f"KV cache type mismatch (file: {sidecar.cache_type_k}/{sidecar.cache_type_v}, "
            f"server: {server.cache_type_k}/{server.cache_type_v})",
        )
    if sidecar.swa_full != server.swa_full:
        refusals.append(
            "swa_full mismatch: restoring an SWA cache without --swa-full is "
            "silently ineffective (measured)",
        )
    if sidecar.n_saved_tokens > server.ctx_size:
        refusals.append(
            f"state does not fit: {sidecar.n_saved_tokens} saved tokens > server "
            f"ctx {server.ctx_size} (ctx-fit rule; server would refuse anyway, loudly)",
        )

    # build tag is advisory only (measured compatible across builds)

    verdict = REFUSE if refusals else (ADVISORY if advisories else OK)
    return Decision(verdict, refusals, advisories)


def check_build(sidecar: Sidecar, server_build_tag: str) -> Decision:
    """Separate advisory-only check for build drift.

    Needs the caller to know the running server's build tag; kept apart from
    `check` so callers without that information still get the full REFUSE
    contract.
    """
    advisories: list[str] = []
    if sidecar.build_tag and server_build_tag and sidecar.build_tag != server_build_tag:
        advisories.append(
            f"llama.cpp build differs (file: {sidecar.build_tag}, server: "
            f"{server_build_tag}): measured compatible across >=5 weeks of releases, "
            "but the format has no upstream stability guarantee",
        )
    return Decision(ADVISORY if advisories else OK, [], advisories)
