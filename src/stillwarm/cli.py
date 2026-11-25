# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""stillwarm CLI.

    stillwarm list    <cache_dir>                 sessions, sizes, ages
    stillwarm inspect <save_file>                 sidecar
    stillwarm prune   <cache_dir> --budget-gb N   LRU-prune to a disk budget
    stillwarm proxy   --cache-dir DIR --server URL ...
                      KV-cache middleware in front of one llama-server

Human output is a rich table by default; every command takes --json for the
raw machine-readable form. Exit codes: 0 success, 1 runtime error, 2 usage.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import click
import httpx
import orjson
import uvicorn
from rich.console import Console
from rich.table import Table

from . import __version__, store
from .proxy import ProxyConfig, ProxySettings, ProxyStartupError, build_app
from .server import ServerError
from .sidecar import Sidecar, sidecar_path_for

_BYTES_PER_GB = 10**9
_SAMPLER_TRUNCATE_AT = 60
console = Console()


def _emit_json(obj: object) -> None:
    """Emit one JSON document (stable format) for machine consumers."""
    payload = orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    click.echo(payload.decode())


def _render_value(value: object) -> str:
    """Render one sidecar field for the human table."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text or "-"


@click.group(name="stillwarm")
@click.version_option(__version__, prog_name="stillwarm")
def cli() -> None:
    """Manage saved llama-server KV-cache sessions."""


@cli.command("list")
@click.argument("cache_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print raw session records as JSON.")
def list_sessions(cache_dir: Path, as_json: bool) -> None:
    """List stored sessions with sizes, ages and sidecar fields."""
    try:
        records = store.sessions(cache_dir)
    except OSError as e:
        msg = f"cannot list {cache_dir}: {e}"
        raise click.ClickException(msg) from e
    if as_json:
        _emit_json(records)
        return
    if not records:
        console.print(f"no sessions in {cache_dir}")
        return
    table = Table(header_style="bold")
    table.add_column("session")
    table.add_column("size (GB)", justify="right")
    table.add_column("age (h)", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("k/fa")
    table.add_column("build")
    total = 0
    for rec in records:
        total += rec["bytes"]
        sidecar = rec.get("sidecar")
        if isinstance(sidecar, dict):
            tokens = str(sidecar.get("n_saved_tokens", "-"))
            kfa = f"{sidecar.get('cache_type_k', '?')}/fa={sidecar.get('flash_attn', '?')}"
            build = str(sidecar.get("build_tag") or "no-tag")
        else:
            tokens = kfa = build = "-"
        table.add_row(
            str(rec["session_id"]),
            f"{rec['bytes'] / _BYTES_PER_GB:.3f}",
            f"{rec['age_hours']:.1f}",
            tokens,
            kfa,
            build,
        )
    console.print(table)
    total_text = f"{len(records)} session(s), {total / _BYTES_PER_GB:.3f} GB"
    console.print(f"[bold]total:[/bold] {total_text}")
    for rec in records:
        if "sidecar_error" in rec:
            console.print(f"[yellow]warning:[/yellow] {rec['session_id']}: {rec['sidecar_error']}")


@cli.command("inspect")
@click.argument("save_file")
@click.option("--json", "as_json", is_flag=True, help="Print the raw sidecar JSON.")
def inspect_file(save_file: str, as_json: bool) -> None:
    """Show the sidecar next to a save file."""
    sidecar_path = sidecar_path_for(save_file)
    if not sidecar_path.exists():
        msg = f"no sidecar for {save_file} (expected {sidecar_path})"
        raise click.ClickException(msg)
    try:
        sidecar = Sidecar.read(sidecar_path)
    except (OSError, ValueError) as e:
        msg = f"cannot read sidecar {sidecar_path}: {e}"
        raise click.ClickException(msg) from e
    if as_json:
        _emit_json(sidecar.to_dict())
        return
    truncated = False
    rows: list[tuple[str, str]] = []
    for name, value in asdict(sidecar).items():
        text = _render_value(value)
        if name == "probe_sampler_json" and len(text) > _SAMPLER_TRUNCATE_AT:
            text = f"{text[:_SAMPLER_TRUNCATE_AT]}..."
            truncated = True
        rows.append((name, text))
    table = Table.grid(padding=(0, 1))
    for row in rows:
        table.add_row(*row)
    console.print(table)
    if truncated:
        console.print("[dim]probe_sampler_json truncated; use --json for the full value[/dim]")


@cli.command("prune")
@click.argument("cache_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--budget-gb", required=True, type=click.FloatRange(min=0), help="Disk budget in GB.")
@click.option("--dry-run", is_flag=True, help="Report what would be removed; delete nothing.")
@click.option("--json", "as_json", is_flag=True, help="Print the raw report JSON.")
def prune_sessions(cache_dir: Path, budget_gb: float, dry_run: bool, as_json: bool) -> None:
    """Prune stored sessions (LRU) until under a disk budget."""
    try:
        report = store.prune(cache_dir, budget_gb, dry_run=dry_run)
    except OSError as e:
        msg = f"cannot prune {cache_dir}: {e}"
        raise click.ClickException(msg) from e
    if as_json:
        _emit_json(report)
        return
    removed = report["removed"]
    skipped = report["skipped_locked"]
    removed_bytes = 0
    for entry in removed:
        removed_bytes += entry["bytes"]
    verb = "would remove" if dry_run else "removed"
    summary = (
        f"{verb} {len(removed)} session(s) ({removed_bytes / _BYTES_PER_GB:.3f} GB); "
        f"{len(skipped)} locked skipped; "
        f"{report['remaining_bytes'] / _BYTES_PER_GB:.3f} GB remain"
    )
    console.print(summary)
    if dry_run:
        console.print("[dim](dry run: nothing deleted)[/dim]")
    if removed:
        table = Table(header_style="bold")
        table.add_column("session")
        table.add_column("size (GB)", justify="right")
        table.add_column("age (h)", justify="right")
        for entry in removed:
            table.add_row(
                str(entry["session_id"]),
                f"{entry['bytes'] / _BYTES_PER_GB:.3f}",
                f"{entry['age_hours']:.1f}",
            )
        console.print(table)
    elif not dry_run:
        console.print("already under budget; nothing removed")
    for entry in skipped:
        console.print(
            f"[yellow]warning:[/yellow] {entry['session_id']} skipped (locked): {entry['error']}"
        )


@cli.command("proxy")
@click.option(
    "--server",
    default=None,
    help="Upstream llama-server base URL (env STILLWARM_BASE_URL).",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory holding the session triples; must equal the upstream --slot-save-path.",
)
@click.option(
    "--model-sha256",
    "model_sha256",
    default=None,
    help="SHA-256 of the model GGUF (env STILLWARM_MODEL_SHA256).",
)
@click.option(
    "--flash-attn",
    "flash_attn",
    type=click.Choice(["on", "off"]),
    default=None,
    help="The upstream flash-attn setting, resolved (env STILLWARM_FLASH_ATTN).",
)
@click.option(
    "--cache-type-k",
    "cache_type_k",
    default=None,
    help="KV cache type for k (env STILLWARM_CACHE_TYPE_K).",
)
@click.option(
    "--cache-type-v",
    "cache_type_v",
    default=None,
    help="KV cache type for v (env STILLWARM_CACHE_TYPE_V).",
)
@click.option(
    "--swa-full",
    "swa_full",
    is_flag=True,
    default=None,
    help="The upstream was started with --swa-full (env STILLWARM_SWA_FULL).",
)
@click.option(
    "--ctx-size",
    "ctx_size",
    type=int,
    default=None,
    help="Slot ctx size; default: taken from the upstream GET /props.",
)
@click.option("--host", default=None, help="Bind address (env STILLWARM_HOST).")
@click.option("--port", type=int, default=None, help="Bind port (env STILLWARM_PORT).")
@click.option(
    "--token",
    default=None,
    help="Bearer token clients must present; never forwarded (env STILLWARM_TOKEN).",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Upstream request timeout in seconds (env STILLWARM_TIMEOUT).",
)
@click.option(
    "--budget-gb",
    "budget_gb",
    type=float,
    default=None,
    help="Disk budget; LRU-prune after each save (env STILLWARM_DISK_BUDGET_GB).",
)
@click.option(
    "--no-probe",
    "no_probe",
    is_flag=True,
    default=None,
    help="Save without the determinism probe (env STILLWARM_NO_PROBE).",
)
def proxy_command(  # noqa: PLR0913, PLR0917 — CLI flags mirror ProxyConfig one-for-one
    server: str | None,
    cache_dir: Path,
    model_sha256: str | None,
    flash_attn: str | None,
    cache_type_k: str | None,
    cache_type_v: str | None,
    swa_full: bool | None,
    ctx_size: int | None,
    host: str | None,
    port: int | None,
    token: str | None,
    timeout: float | None,
    budget_gb: float | None,
    no_probe: bool | None,
) -> None:
    """Run the KV-cache middleware in front of one single-slot llama-server."""
    settings = ProxySettings()
    cfg = ProxyConfig(
        base_url=server or settings.base_url,
        cache_dir=str(cache_dir),
        model_sha256=model_sha256 or settings.model_sha256,
        flash_attn=flash_attn or settings.flash_attn,
        cache_type_k=cache_type_k or settings.cache_type_k,
        cache_type_v=cache_type_v or settings.cache_type_v,
        swa_full=swa_full if swa_full is not None else settings.swa_full,
        ctx_size=ctx_size if ctx_size is not None else settings.ctx_size,
        host=host or settings.host,
        port=port if port is not None else settings.port,
        token=token if token is not None else settings.token,
        timeout=timeout if timeout is not None else settings.timeout,
        disk_budget_gb=budget_gb if budget_gb is not None else settings.disk_budget_gb,
        no_probe=no_probe if no_probe is not None else settings.no_probe,
    )
    missing = [
        label
        for label, value in (
            ("--model-sha256 (STILLWARM_MODEL_SHA256)", cfg.model_sha256),
            ("--flash-attn (STILLWARM_FLASH_ATTN)", cfg.flash_attn),
            ("--cache-type-k (STILLWARM_CACHE_TYPE_K)", cfg.cache_type_k),
            ("--cache-type-v (STILLWARM_CACHE_TYPE_V)", cfg.cache_type_v),
        )
        if not value
    ]
    if missing:
        msg = "stillwarm proxy requires: " + ", ".join(missing)
        raise click.ClickException(msg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        app = build_app(cfg)
    except (ProxyStartupError, ServerError, httpx.RequestError) as e:
        msg = f"cannot start proxy: {e}"
        raise click.ClickException(msg) from e
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=60,
    )


def main() -> int:
    """Run the stillwarm command-line interface."""
    cli.main()
    return 0
