"""stillwarm CLI.

    stillwarm list    <cache_dir>              sessions, sizes, ages
    stillwarm inspect <save_file>              sidecar + cache key
    stillwarm prune   <cache_dir> --budget-gb N [--dry-run]

The session wrapper itself is a library API (stillwarm.Session) — wiring it to a
long-running proxy CLI is deliberately out of v0.1 scope.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, store
from .sidecar import Sidecar, sidecar_path_for


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stillwarm",
                                 description="KV-cache session persistence for llama-server")
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
        total = sum(r["bytes"] for r in recs)
        for r in recs:
            sc = r.get("sidecar") or {}
            print(f"{r['session_id']:<28} {r['bytes']/1e9:8.3f} GB  age {r['age_hours']:>7.1f} h  "
                  f"tokens {sc.get('n_saved_tokens', '?'):>7}  "
                  f"[{sc.get('cache_type_k', '?')}/fa={sc.get('flash_attn', '?')}"
                  f"/{sc.get('build_tag') or 'no-tag'}]")
        print(f"{'TOTAL':<28} {total/1e9:8.3f} GB  ({len(recs)} sessions)")
        return 0

    if ns.cmd == "inspect":
        sc = Sidecar.read(sidecar_path_for(ns.save_file))
        print(json.dumps(sc.to_dict(), indent=2, sort_keys=True))
        print(f"cache_key: {sc.cache_key()}", file=sys.stderr)
        return 0

    if ns.cmd == "prune":
        report = store.prune(ns.cache_dir, ns.budget_gb, dry_run=ns.dry_run)
        print(json.dumps(report, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
