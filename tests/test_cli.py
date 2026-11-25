"""Contract pins for the stillwarm CLI: list/inspect/prune must print their
documented output (sessions + total, sidecar JSON, prune JSON report)."""

import json

from stillwarm import cli
from stillwarm.sidecar import Sidecar


def _mk_store(d, sid="s1", size=100) -> None:
    (d / f"{sid}.bin").write_bytes(b"\0" * size)
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
    ).write(d / f"{sid}.bin.stillwarm.json")


def test_list_prints_sessions_and_total(tmp_path, capsys) -> None:
    _mk_store(tmp_path)
    code = cli.main(["list", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "s1" in out
    assert "TOTAL" in out
    assert "1 sessions" in out


def test_list_empty_dir_prints_zero_total(tmp_path, capsys) -> None:
    code = cli.main(["list", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "0 sessions" in out


def test_inspect_prints_sidecar_json(tmp_path, capsys) -> None:
    _mk_store(tmp_path)
    code = cli.main(["inspect", str(tmp_path / "s1.bin")])
    out = capsys.readouterr().out
    assert code == 0
    d = json.loads(out)
    assert d["model_sha256"] == "m" * 64
    assert d["n_saved_tokens"] == 10


def test_prune_dry_run_prints_report(tmp_path, capsys) -> None:
    _mk_store(tmp_path)
    code = cli.main(["prune", str(tmp_path), "--budget-gb", "0", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    report = json.loads(out)
    assert report["dry_run"] is True
    assert report["removed"][0]["session_id"] == "s1"
    assert (tmp_path / "s1.bin").exists()  # dry run deletes nothing
