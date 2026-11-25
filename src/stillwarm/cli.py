# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""stillwarm CLI.

    stillwarm list    <cache_dir>              sessions, sizes, ages
    stillwarm inspect <save_file>              sidecar + cache key
    stillwarm prune   <cache_dir> --budget-gb N [--dry-run]

The session wrapper itself is a library API (stillwarm.Session) — wiring it to a
long-running proxy CLI is deliberately out of v0.1 scope.
"""

from __future__ import annotations

import argparse

from . import __version__, store
from .sidecar import Sidecar, sidecar_path_for


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for stillwarm session management."""
    ap = argparse.ArgumentParser(
        prog="stillwarm",
        description="KV-cache session persistence for llama-server",
    )
    ap.add_argument("--version", action="version", version=f"stillwarm {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list stored sessions (sizes, ages)")
    p_list.add_argument("cache_dir")

    p_inspect = sub.add_parser("inspect", help="print the sidecar for a save file")
    p_inspect.add_argument("save_file")

    p_prune = sub.add_parser("prune", help="LRU-prune sessions to a disk budget")
    p_prune.add_argument("cache_dir")
    p_prune.add_argument("--budget-gb", type=float, required=True)
    p_prune.add_argument("--dry-run", action="store_true")

    ns = ap.parse_args(argv)

    if ns.cmd == "list":
        recs = store.sessions(ns.cache_dir)
        sum(r["bytes"] for r in recs)
        for r in recs:
            r.get("sidecar") or {}
        return 0

    if ns.cmd == "inspect":
        Sidecar.read(sidecar_path_for(ns.save_file))
        return 0

    if ns.cmd == "prune":
        store.prune(ns.cache_dir, ns.budget_gb, dry_run=ns.dry_run)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
