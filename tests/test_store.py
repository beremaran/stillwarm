import os
import time
from pathlib import Path
from typing import Any

import pytest

from stillwarm import store
from stillwarm.sidecar import Sidecar

_BYTES_PER_GB = 1e9


def _mk_session(d: Path, sid: str, size: int, age_s: float) -> None:
    b = d / f"{sid}.bin"
    b.write_bytes(b"\0" * size)
    Sidecar(
        model_sha256="m" * 64,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        n_saved_tokens=10,
        prompt_prefix_sha256="p" * 64,
        save_file=b.name,
        size_bytes=size,
    ).write(d / f"{sid}.bin.stillwarm.json")
    (d / f"{sid}.prefix.txt").write_text("prefix")
    t = time.time() - age_s
    os.utime(b, (t, t))


def test_list_sorted_oldest_first(tmp_path: Path) -> None:
    _mk_session(tmp_path, "new", 100, age_s=10)
    _mk_session(tmp_path, "old", 100, age_s=1000)
    recs = store.sessions(tmp_path)
    assert [r["session_id"] for r in recs] == ["old", "new"]
    assert recs[0]["sidecar"]["n_saved_tokens"] == 10


def test_prune_lru_removes_triple(tmp_path: Path) -> None:
    _mk_session(tmp_path, "old", 600, age_s=1000)
    _mk_session(tmp_path, "new", 600, age_s=10)
    rep = store.prune(tmp_path, budget_gb=1000 / _BYTES_PER_GB)  # budget = 1000 bytes
    assert [r["session_id"] for r in rep["removed"]] == ["old"]
    assert not (tmp_path / "old.bin").exists()
    assert not (tmp_path / "old.bin.stillwarm.json").exists()
    assert not (tmp_path / "old.prefix.txt").exists()
    assert (tmp_path / "new.bin").exists()  # newest survives


def test_prune_dry_run(tmp_path: Path) -> None:
    _mk_session(tmp_path, "a", 600, age_s=100)
    rep = store.prune(tmp_path, budget_gb=1 / _BYTES_PER_GB, dry_run=True)  # budget = 1 byte
    assert rep["removed"]
    assert (tmp_path / "a.bin").exists()


def test_prune_under_budget_noop(tmp_path: Path) -> None:
    _mk_session(tmp_path, "a", 100, age_s=100)
    assert store.prune(tmp_path, budget_gb=1.0)["removed"] == []


def test_prune_tolerates_locked_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows holds open files: PermissionError on unlink must skip the locked
    session and keep pruning others (Windows holds open files).
    """
    _mk_session(tmp_path, "locked", 600, age_s=2000)
    _mk_session(tmp_path, "old", 600, age_s=1000)
    _mk_session(tmp_path, "new", 600, age_s=10)
    real_unlink = Path.unlink

    def fake_unlink(self: Path, *a: Any, **k: Any) -> None:
        if "locked" in self.name:
            raise PermissionError(13, "file in use", str(self))
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    rep = store.prune(tmp_path, budget_gb=1300 / _BYTES_PER_GB)  # room for 2 files
    assert [s["session_id"] for s in rep["skipped_locked"]] == ["locked"]
    assert [r["session_id"] for r in rep["removed"]] == ["old"]
    assert (tmp_path / "locked.bin").exists()  # untouched
    assert not (tmp_path / "old.bin").exists()  # pruned instead
    assert (tmp_path / "new.bin").exists()  # newest kept
