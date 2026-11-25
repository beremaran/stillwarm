from stillwarm import ADVISORY, OK, REFUSE, ServerProfile, check, check_build
from stillwarm.sidecar import Sidecar


def sc(**kw):
    base = dict(
        model_sha256="m" * 64,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        n_saved_tokens=8193,
        prompt_prefix_sha256="p" * 64,
        build_tag="b9871",
    )
    base.update(kw)
    return Sidecar(**base)


def srv(**kw):
    base = dict(
        model_sha256="m" * 64,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        ctx_size=16384,
    )
    base.update(kw)
    return ServerProfile(**base)


def test_ok():
    d = check(sc(), srv())
    assert d.verdict == OK and bool(d)


def test_fa_mismatch_refused():  # Block D d5: FAILS-CLEAN both ways
    d = check(sc(), srv(flash_attn="off"))
    assert d.verdict == REFUSE and not bool(d)
    assert any("flash-attention" in r for r in d.refusals)


def test_cache_type_mismatch_refused():
    assert check(sc(), srv(cache_type_k="q8_0", cache_type_v="q8_0")).verdict == REFUSE


def test_swa_full_mismatch_refused():  # Block E exhibit: SILENTLY-INEFFECTIVE
    assert check(sc(swa_full=True), srv(swa_full=False)).verdict == REFUSE


def test_model_mismatch_refused():
    assert check(sc(model_sha256="x" * 64), srv()).verdict == REFUSE


def test_ctx_fit_rule():  # Block D d6 addendum
    assert check(sc(n_saved_tokens=9000), srv(ctx_size=8192)).verdict == REFUSE
    assert check(sc(n_saved_tokens=4097), srv(ctx_size=8192)).verdict == OK


def test_build_drift_advisory_only():  # Block D d3: WORKS both directions
    d = check_build(sc(build_tag="b9386"), "b9871")
    assert d.verdict == ADVISORY and bool(d)
    assert check_build(sc(build_tag="b9871"), "b9871").verdict == OK
