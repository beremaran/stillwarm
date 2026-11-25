"""Integration tests against a real llama-server + a tiny GGUF (<1 GB).

Covers the measured Block-D taxonomy end-to-end: WORKS, REFUSE(fa),
REFUSE(ctx-fit), ADVISORY(build), verify-pass, verify-mismatch (corrupted copy),
and the SILENTLY-INEFFECTIVE guard path.

Environment (skipped entirely if absent):
  STILLWARM_TEST_SERVER_BIN  path to llama-server
  STILLWARM_TEST_MODEL       path to the tiny GGUF
Defaults match the stillwarm-bench pinned setup (b9871 + SmolLM2-135M Q8_0,
sha256 recorded in stillwarm-bench/env/models.lock).
"""

import os
import shutil
import subprocess
import time
import warnings

import pytest

from stillwarm import (
    IneffectiveRestoreWarning,
    LlamaServer,
    RefusedRestoreError,
    ServerProfile,
    Session,
    VerifyFailedError,
)
from stillwarm.sidecar import Sidecar, sidecar_path_for

BIN = os.environ.get(
    "STILLWARM_TEST_SERVER_BIN",
    "/Users/vimal/llamacpp-stillwarm/build/bin/llama-server",
)
MODEL = os.environ.get(
    "STILLWARM_TEST_MODEL",
    "/Users/vimal/models/SmolLM2-135M-Instruct-Q8_0.gguf",
)
MODEL_SHA = "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"
PORT = 8213
BUILD_TAG = "b9871"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(BIN) and os.path.exists(MODEL)),
    reason="llama-server binary or tiny test model not available",
)

PREFIX = (
    "The five essential rules of the fictional Order of the Quiet Lantern are: "
    "First, never light two lanterns in one window. Second, always thank the "
    "lamplighter by name. Third, oil is counted on Mondays. Fourth, a broken "
    "wick must be buried at dusk. Fifth, the youngest keeper holds the key. "
)
QUESTION = "Question: What happens on Mondays?\nAnswer:"


class Server:
    """Test-scoped llama-server lifecycle."""

    def __init__(self, ctx=1024, fa="on", extra=()) -> None:
        self.proc = None
        self.ctx = ctx
        self.args = [
            BIN,
            "-m",
            MODEL,
            "-c",
            str(ctx),
            "-np",
            "1",
            "-ngl",
            "99",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--seed",
            "42",
            "-fa",
            fa,
            "--cache-ram",
            "0",
            "--slot-save-path",
            self.save_dir,
            *list(extra),
        ]

    save_dir = ""  # set per-test via fixture

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        api = LlamaServer(f"http://127.0.0.1:{PORT}", timeout=120)
        for _ in range(200):
            if api.health():
                return api
            if self.proc.poll() is not None:
                msg = "test server died on startup"
                raise RuntimeError(msg)
            time.sleep(0.15)
        msg = "test server not ready"
        raise TimeoutError(msg)

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture
def cache_dir(tmp_path):
    Server.save_dir = str(tmp_path) + "/"
    return tmp_path


def profile(fa="on", ctx=1024):
    return ServerProfile(
        model_sha256=MODEL_SHA,
        flash_attn=fa,
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        ctx_size=ctx,
    )


def make_session(api, cache_dir, **kw):
    return Session(
        api,
        profile(**{k: v for k, v in kw.items() if k in ("fa", "ctx")}),
        cache_dir,
        server_build_tag=kw.get("build_tag", BUILD_TAG),
        disk_budget_gb=kw.get("disk_budget_gb"),
    )


def test_works_roundtrip_across_restart(cache_dir) -> None:
    """WORKS: ask -> (server dies) -> ask again; restore reuses the prefix."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        r1 = sw.ask(
            "s1",
            PREFIX + QUESTION,
            n_predict=24,
            sampler={"temperature": 0, "top_k": 1, "seed": 42, "ignore_eos": True},
        )
        assert not r1.restored
        assert r1.content
    with Server() as api:  # fresh process
        sw = make_session(api, cache_dir)
        r2 = sw.ask("s1", PREFIX + "Question: Who holds the key?\nAnswer:", n_predict=24)
        assert r2.restored
        assert r2.warning == ""  # guard silent
        # reuse evidence: only the new tail was prefilled
        assert r2.prompt_n is not None
        assert r2.prompt_n < 64


def test_refuse_fa_mismatch(cache_dir) -> None:
    """REFUSE(fa): sidecar fa=on must not load into an fa=off profile."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask("s1", PREFIX + QUESTION, n_predict=8)
    with Server(fa="off") as api:
        sw = Session(api, profile(fa="off"), cache_dir, server_build_tag=BUILD_TAG)
        with pytest.raises(RefusedRestoreError, match="flash-attention"):
            sw.restore("s1")


def test_refuse_ctx_fit(cache_dir) -> None:
    """REFUSE(ctx-fit): state saved at ctx 1024 must not load into ctx smaller
    than the state (sidecar n_saved_tokens > server ctx).
    """
    with Server(ctx=1024) as api:
        sw = make_session(api, cache_dir)
        sw.ask("s1", PREFIX * 6 + QUESTION, n_predict=8)  # ~400+ tokens state
        n_saved = Sidecar.read(sidecar_path_for(cache_dir / "s1.bin")).n_saved_tokens
        assert n_saved > 256
    with Server(ctx=256) as api:
        sw = Session(api, profile(ctx=256), cache_dir, server_build_tag=BUILD_TAG)
        with pytest.raises(RefusedRestoreError, match="does not fit"):
            sw.restore("s1")


def test_advisory_build_tag(cache_dir) -> None:
    """ADVISORY(build): different build tag warns but restores fine."""
    with Server() as api:
        sw = make_session(api, cache_dir, build_tag="b0000")  # saved under 'b0000'
        sw.ask("s1", PREFIX + QUESTION, n_predict=8)
    with Server() as api:
        sw = make_session(api, cache_dir, build_tag="b9871")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sw.restore("s1")
        assert any("build differs" in str(x.message) for x in w)


def test_verify_pass(cache_dir) -> None:
    """--verify: restored state reproduces the save-time probe hash."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask("s1", PREFIX + QUESTION, n_predict=24)
    with Server() as api:
        sw = make_session(api, cache_dir)
        assert sw.restore("s1", verify=True) is True
        # post-verify state is the RESTORED state: next ask reuses the prefix
        r = sw.ask("s1", PREFIX + "Question: Who is thanked?\nAnswer:", n_predict=16)
        assert r.prompt_n is not None
        assert r.prompt_n < 64


def test_verify_mismatch_on_corrupted_copy(cache_dir) -> None:
    """--verify catches a corrupted save file (bytes flipped mid-payload)."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask("s1", PREFIX + QUESTION, n_predict=24)
    # corrupt a COPY, keeping the original intact
    src = cache_dir / "s1.bin"
    dst = cache_dir / "s1corrupt.bin"
    shutil.copy(src, dst)
    for aux in ("s1.bin.stillwarm.json", "s1.prefix.txt"):
        shutil.copy(cache_dir / aux, cache_dir / aux.replace("s1", "s1corrupt", 1))
    blob = bytearray(dst.read_bytes())
    mid = len(blob) // 2
    for i in range(mid, mid + 4096):  # stomp 4 KiB of KV payload
        blob[i] ^= 0xFF
    dst.write_bytes(blob)
    with Server() as api:
        sw = make_session(api, cache_dir)
        with pytest.raises((VerifyFailedError, RefusedRestoreError, Exception)) as ei:
            ok = sw.restore("s1corrupt", verify=True)
            assert ok is not True  # if no exception, must not pass
        assert not isinstance(ei.value, AssertionError)


def test_ineffective_guard_fires_on_nonmatching_prompt(cache_dir) -> None:
    """Guard: restore then ask with a DIFFERENT prefix -> near-full re-prefill
    -> IneffectiveRestoreWarning (the guard doesn't care WHY reuse failed).
    """
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask("s1", PREFIX * 3 + QUESTION, n_predict=8)
    with Server() as api:
        sw = make_session(api, cache_dir)
        other = (
            "Completely different text about shipping logistics, containers and harbors. "
        ) * 12 + "Question: what is this about?\nAnswer:"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = sw.ask("s1", other, n_predict=8)
        assert r.restored
        assert any(isinstance(x.message, IneffectiveRestoreWarning) for x in w)
        assert "SILENTLY INEFFECTIVE" in r.warning


def test_lru_prune_via_session(cache_dir) -> None:
    """Disk budget: oldest session pruned when over budget."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask("old", PREFIX + QUESTION, n_predict=8)
        time.sleep(1.1)
        one = (cache_dir / "old.bin").stat().st_size
        sw.disk_budget_gb = 1.5 * one / 1e9  # fits ONE state, not two
        sw.ask("new", PREFIX + "Question: Rule one?\nAnswer:", n_predict=8)
        assert not (cache_dir / "old.bin").exists()
        assert (cache_dir / "new.bin").exists()


def test_save_point_after_generation_in_ask_flow(cache_dir) -> None:
    """Regression: ask() generates a reply, so the saved state ends
    with sampled tokens -> sidecar must say after_generation (was mislabeled).
    """
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask(
            "s1",
            PREFIX + QUESTION,
            n_predict=16,
            sampler={"temperature": 0, "top_k": 1, "seed": 42, "ignore_eos": True},
        )
        sc = Sidecar.read(sidecar_path_for(cache_dir / "s1.bin"))
        assert sc.save_point == "after_generation"
        # numeric cross-check: state is larger than the pre-generation prompt
        pre_gen = len(api.tokenize(PREFIX + QUESTION))
        assert sc.n_saved_tokens > pre_gen


def test_save_point_doc_prefill_in_prime_flow(cache_dir) -> None:
    """Prime flow: prefill-only state (no sampled tokens) -> doc_prefill."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        api.completion(PREFIX, 0, cache_prompt=True)  # prefill, generate nothing
        sw._live_prefix["primed"] = PREFIX
        sc = sw.save("primed")  # generated_tokens defaults to 0
        assert sc.save_point == "doc_prefill"
        assert sc.n_saved_tokens <= len(api.tokenize(PREFIX)) + 2  # prompt(+BOS) only


def test_guard_message_mentions_after_generation(cache_dir) -> None:
    """Guard must use the (now-correct) save_point in its diagnosis."""
    with Server() as api:
        sw = make_session(api, cache_dir)
        sw.ask("s1", PREFIX * 3 + QUESTION, n_predict=8)  # after_generation state
    with Server() as api:
        sw = make_session(api, cache_dir)
        other = (
            "Entirely unrelated inventory report about warehouse pallets. "
        ) * 14 + "Question: topic?\nAnswer:"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = sw.ask("s1", other, n_predict=8)
        assert "save_point=after_generation" in r.warning
