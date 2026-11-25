import json

import pytest

from stillwarm import Sidecar, sidecar_path_for
from stillwarm.sidecar import SIDECAR_VERSION, IncompatibleSidecar


def mk(**kw):
    base = dict(
        model_sha256="m" * 64,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        n_saved_tokens=8193,
        prompt_prefix_sha256="p" * 64,
    )
    base.update(kw)
    return Sidecar(**base)


def test_roundtrip(tmp_path):
    sc = mk(build_tag="b9871", probe_hash="h" * 64, save_file="x.bin", size_bytes=1)
    p = sc.write(tmp_path / "x.bin.stillwarm.json")
    assert Sidecar.read(p) == sc


def test_auto_fa_rejected():
    with pytest.raises(ValueError, match="auto"):
        mk(flash_attn="auto")


def test_unknown_version_refused(tmp_path):
    sc = mk()
    d = sc.to_dict()
    d["sidecar_version"] = SIDECAR_VERSION + 1
    p = tmp_path / "s.json"
    p.write_text(json.dumps(d))
    with pytest.raises(IncompatibleSidecar):
        Sidecar.read(p)


def test_cache_key_components():
    a = mk()
    assert a.cache_key() == mk().cache_key()  # deterministic
    assert a.cache_key() != mk(flash_attn="off").cache_key()  # fa in key
    assert a.cache_key() != mk(cache_type_k="q8_0").cache_key()  # ctk in key
    assert a.cache_key() != mk(swa_full=True).cache_key()  # swa_full in key
    assert a.cache_key() != mk(prompt_prefix_sha256="q" * 64).cache_key()
    # measured-irrelevant properties are NOT part of the key:
    assert a.cache_key() == mk(build_tag="b9386").cache_key()  # build advisory only
    assert a.cache_key() == mk(n_saved_tokens=42).cache_key()  # ctx handled by fit rule


def test_sidecar_path():
    assert str(sidecar_path_for("/x/y/cache.bin")).endswith("/x/y/cache.bin.stillwarm.json")
