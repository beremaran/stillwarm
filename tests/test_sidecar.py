import json

import pytest

from stillwarm import Sidecar, sidecar_path_for
from stillwarm.sidecar import SIDECAR_VERSION, IncompatibleSidecarError


def mk(**kw):
    base = {
        "model_sha256": "m" * 64,
        "flash_attn": "on",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "swa_full": False,
        "n_saved_tokens": 8193,
        "prompt_prefix_sha256": "p" * 64,
    }
    base.update(kw)
    return Sidecar(**base)


def test_roundtrip(tmp_path) -> None:
    sc = mk(build_tag="b9871", probe_hash="h" * 64, save_file="x.bin", size_bytes=1)
    p = sc.write(tmp_path / "x.bin.stillwarm.json")
    assert Sidecar.read(p) == sc


def test_auto_fa_rejected() -> None:
    with pytest.raises(ValueError, match="auto"):
        mk(flash_attn="auto")


def test_unknown_version_refused(tmp_path) -> None:
    sc = mk()
    d = sc.to_dict()
    d["sidecar_version"] = SIDECAR_VERSION + 1
    p = tmp_path / "s.json"
    p.write_text(json.dumps(d))
    with pytest.raises(IncompatibleSidecarError):
        Sidecar.read(p)


def test_sidecar_path() -> None:
    assert str(sidecar_path_for("/x/y/cache.bin")).endswith("/x/y/cache.bin.stillwarm.json")
