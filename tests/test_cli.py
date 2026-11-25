"""CLI contract pins: list/inspect/prune human output, --json shapes, exit codes."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from stillwarm.cli import cli
from stillwarm.sidecar import Sidecar, sidecar_path_for


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"COLUMNS": "200"})


def _mk_session(d: Path, sid: str = "s1", size: int = 100) -> Path:
    bin_path = d / f"{sid}.bin"
    bin_path.write_bytes(b"\0" * size)
    Sidecar(
        model_sha256="m" * 64,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        n_saved_tokens=10,
        prompt_prefix_sha256="p" * 64,
        build_tag="b9871",
        save_file=f"{sid}.bin",
        size_bytes=size,
    ).write(sidecar_path_for(bin_path))
    return bin_path


def test_list_human_shows_session_and_total(runner: CliRunner, tmp_path: Path) -> None:
    _mk_session(tmp_path)
    result = runner.invoke(cli, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "s1" in result.output
    assert "f16/fa=on" in result.output
    assert "b9871" in result.output
    assert "total:" in result.output
    assert "1 session(s)" in result.output


def test_list_json_records(runner: CliRunner, tmp_path: Path) -> None:
    _mk_session(tmp_path)
    result = runner.invoke(cli, ["list", str(tmp_path), "--json"])
    assert result.exit_code == 0
    records = json.loads(result.output)
    assert records[0]["session_id"] == "s1"
    assert records[0]["bytes"] == 100
    assert records[0]["sidecar"]["n_saved_tokens"] == 10


def test_list_empty_dir(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "no sessions" in result.output


def test_inspect_human_shows_fields(runner: CliRunner, tmp_path: Path) -> None:
    _mk_session(tmp_path)
    result = runner.invoke(cli, ["inspect", str(tmp_path / "s1.bin")])
    assert result.exit_code == 0
    assert "model_sha256" in result.output
    assert "swa_full" in result.output
    assert "false" in result.output
    assert "b9871" in result.output


def test_inspect_json_sidecar(runner: CliRunner, tmp_path: Path) -> None:
    _mk_session(tmp_path)
    result = runner.invoke(cli, ["inspect", str(tmp_path / "s1.bin"), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["model_sha256"] == "m" * 64
    assert data["n_saved_tokens"] == 10


def test_inspect_missing_sidecar_exit_1(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["inspect", str(tmp_path / "ghost.bin")])
    assert result.exit_code == 1
    assert "no sidecar" in result.output


def test_inspect_corrupt_sidecar_exit_1(runner: CliRunner, tmp_path: Path) -> None:
    bin_path = _mk_session(tmp_path)
    sidecar_path_for(bin_path).write_bytes(b"not json")
    result = runner.invoke(cli, ["inspect", str(bin_path)])
    assert result.exit_code == 1
    assert "cannot read sidecar" in result.output


def test_prune_dry_run_deletes_nothing(runner: CliRunner, tmp_path: Path) -> None:
    _mk_session(tmp_path)
    result = runner.invoke(cli, ["prune", str(tmp_path), "--budget-gb", "0", "--dry-run", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["dry_run"] is True
    assert report["removed"][0]["session_id"] == "s1"
    assert (tmp_path / "s1.bin").exists()


def test_prune_removes_oldest_triple(runner: CliRunner, tmp_path: Path) -> None:
    old = _mk_session(tmp_path, sid="old", size=1500)
    new = _mk_session(tmp_path, sid="new", size=100)
    os.utime(old, (1_000_000, 1_000_000))  # mtime drives LRU order
    result = runner.invoke(cli, ["prune", str(tmp_path), "--budget-gb", "0.000001"])
    assert result.exit_code == 0
    assert "removed 1 session(s)" in result.output
    assert not old.exists()
    assert not sidecar_path_for(old).exists()
    assert not (tmp_path / "old.prefix.txt").exists()
    assert new.exists()


def test_prune_missing_dir_exit_2(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["prune", str(tmp_path / "nope"), "--budget-gb", "1"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_prune_negative_budget_exit_2(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["prune", str(tmp_path), "--budget-gb", "-1"])
    assert result.exit_code == 2


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.4.0" in result.output
