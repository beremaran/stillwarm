import os, time
from stillwarm import store
from stillwarm.sidecar import Sidecar


def _mk_session(d, sid, size, age_s):
    b = d / f"{sid}.bin"
    b.write_bytes(b"\0" * size)
    Sidecar(model_sha256="m"*64, flash_attn="on", cache_type_k="f16", cache_type_v="f16",
            swa_full=False, n_saved_tokens=10, prompt_prefix_sha256="p"*64,
            save_file=b.name, size_bytes=size).write(d / f"{sid}.bin.stillwarm.json")
    (d / f"{sid}.prefix.txt").write_text("prefix")
    t = time.time() - age_s
    os.utime(b, (t, t))


def test_list_sorted_oldest_first(tmp_path):
    _mk_session(tmp_path, "new", 100, age_s=10)
    _mk_session(tmp_path, "old", 100, age_s=1000)
    recs = store.sessions(tmp_path)
    assert [r["session_id"] for r in recs] == ["old", "new"]
    assert recs[0]["sidecar"]["n_saved_tokens"] == 10


def test_prune_lru_removes_triple(tmp_path):
    _mk_session(tmp_path, "old", 600, age_s=1000)
    _mk_session(tmp_path, "new", 600, age_s=10)
    rep = store.prune(tmp_path, budget_gb=1e-6)          # budget = 1000 bytes
    assert [r["session_id"] for r in rep["removed"]] == ["old"]
    assert not (tmp_path / "old.bin").exists()
    assert not (tmp_path / "old.bin.stillwarm.json").exists()
    assert not (tmp_path / "old.prefix.txt").exists()
    assert (tmp_path / "new.bin").exists()               # newest survives


def test_prune_dry_run(tmp_path):
    _mk_session(tmp_path, "a", 600, age_s=100)
    rep = store.prune(tmp_path, budget_gb=1e-9, dry_run=True)
    assert rep["removed"] and (tmp_path / "a.bin").exists()


def test_prune_under_budget_noop(tmp_path):
    _mk_session(tmp_path, "a", 100, age_s=100)
    assert store.prune(tmp_path, budget_gb=1.0)["removed"] == []


def test_prune_tolerates_locked_files(tmp_path, monkeypatch):
    """Windows holds open files: PermissionError on unlink must skip the locked
    session and keep pruning others (Gate-4.1 Windows audit)."""
    from pathlib import Path
    _mk_session(tmp_path, "locked", 600, age_s=2000)
    _mk_session(tmp_path, "old", 600, age_s=1000)
    _mk_session(tmp_path, "new", 600, age_s=10)
    real_unlink = Path.unlink

    def fake_unlink(self, *a, **k):
        if "locked" in self.name:
            raise PermissionError(13, "file in use", str(self))
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    rep = store.prune(tmp_path, budget_gb=1300e-9)         # room for 2 files: one removal suffices
    assert [s["session_id"] for s in rep["skipped_locked"]] == ["locked"]
    assert [r["session_id"] for r in rep["removed"]] == ["old"]
    assert (tmp_path / "locked.bin").exists()              # untouched
    assert not (tmp_path / "old.bin").exists()             # pruned instead
    assert (tmp_path / "new.bin").exists()                 # newest kept
