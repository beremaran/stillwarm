# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""Session store: list, inspect, and LRU-prune saved sessions under a disk budget.

A stored session = <id>.bin (the KV state) + <id>.bin.stillwarm.json (sidecar)
+ <id>.prefix.txt (retained prefix text, needed for --verify probes).
Pruning removes least-recently-USED sessions first (atime of use = bin mtime,
updated on every save) and always removes the whole triple together.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .sidecar import Sidecar, sidecar_path_for


def _triple(bin_path: Path) -> list[Path]:
    return [
        bin_path,
        sidecar_path_for(bin_path),
        bin_path.with_suffix("")
        .with_suffix("")
        .with_name(bin_path.name.replace(".bin", ".prefix.txt")),
    ]


def sessions(cache_dir: str) -> list[dict[str, Any]]:
    """One record per stored session, oldest-used first."""
    cache_path = Path(str(cache_dir)).expanduser()
    out: list[dict[str, Any]] = []
    for bin_path in sorted(cache_path.glob("*.bin")):
        rec = {
            "session_id": bin_path.name[:-4],
            "bin": str(bin_path),
            "bytes": bin_path.stat().st_size,
            "mtime": bin_path.stat().st_mtime,
            "age_hours": round((time.time() - bin_path.stat().st_mtime) / 3600, 2),
            "sidecar": None,
        }
        sc_path = sidecar_path_for(bin_path)
        if sc_path.exists():
            try:
                rec["sidecar"] = Sidecar.read(sc_path).to_dict()
            except (OSError, ValueError) as e:
                rec["sidecar_error"] = str(e)
        out.append(rec)
    out.sort(key=lambda r: r["mtime"])
    return out


def prune(cache_dir: str, budget_gb: float, *, dry_run: bool = False) -> dict[str, Any]:
    """Delete least-recently-used session triples until under budget_gb.

    Windows note: files held open by another process raise PermissionError on
    unlink. A locked session is SKIPPED (reported, not fatal) and pruning moves
    on to the next-oldest candidate.
    """
    budget = int(budget_gb * 10**9)
    recs = sessions(cache_dir)
    total = sum(r["bytes"] for r in recs)
    removed, skipped_locked = [], []
    for r in recs:
        if total <= budget:
            break
        if dry_run:
            removed.append(
                {"session_id": r["session_id"], "bytes": r["bytes"], "age_hours": r["age_hours"]},
            )
            total -= r["bytes"]
            continue
        try:
            for p in _triple(Path(r["bin"])):
                if p.exists():
                    p.unlink()
        except PermissionError as e:
            skipped_locked.append({"session_id": r["session_id"], "error": str(e)})
            continue
        removed.append(
            {"session_id": r["session_id"], "bytes": r["bytes"], "age_hours": r["age_hours"]},
        )
        total -= r["bytes"]
    return {
        "budget_gb": budget_gb,
        "removed": removed,
        "skipped_locked": skipped_locked,
        "remaining_bytes": total,
        "dry_run": dry_run,
    }
