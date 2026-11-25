"""Integration tests against real llama-server processes (Docker Compose).

Covers the measured Block-D taxonomy end-to-end: WORKS, REFUSE(fa),
REFUSE(ctx-fit), ADVISORY(build), verify-pass, verify-mismatch (corrupted
copy), and the SILENTLY-INEFFECTIVE guard path.

The servers are provided by docker-compose.integration.yml: four
llama-server containers sharing one save volume. The "twin" is a fresh
process that plays "the server restarted". The test container talks to the
servers over HTTP only and never touches the docker daemon.

Environment (all required; the module skips with the run command if any is
missing):
  STILLWARM_TEST_URL_PRIMARY    ctx 1024, flash-attn on
  STILLWARM_TEST_URL_TWIN       ctx 1024, flash-attn on, fresh process
  STILLWARM_TEST_URL_FA_OFF     ctx 1024, flash-attn off
  STILLWARM_TEST_URL_SMALL_CTX  ctx 256, flash-attn on
  STILLWARM_TEST_CACHE_DIR      directory shared with the servers'
                                --slot-save-path

Run:
  docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit test
"""

import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from stillwarm import (
    PROBE_SAMPLER,
    IneffectiveRestoreWarning,
    LlamaServer,
    RefusedRestoreError,
    ServerProfile,
    Session,
    VerifyFailedError,
)
from stillwarm.sidecar import Sidecar, sidecar_path_for

URL_PRIMARY = os.environ.get("STILLWARM_TEST_URL_PRIMARY", "")
URL_TWIN = os.environ.get("STILLWARM_TEST_URL_TWIN", "")
URL_FA_OFF = os.environ.get("STILLWARM_TEST_URL_FA_OFF", "")
URL_SMALL_CTX = os.environ.get("STILLWARM_TEST_URL_SMALL_CTX", "")
CACHE_DIR = os.environ.get("STILLWARM_TEST_CACHE_DIR", "")

# sha256 of the model the compose file globs from the Hugging Face cache
# (bartowski/SmolLM2-135M-Instruct-GGUF, SmolLM2-135M-Instruct-Q8_0.gguf).
MODEL_SHA = "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"
FULL_CTX = 1024
SMALL_CTX = 256
# Tag a save is recorded under in the advisory test; it can never equal the
# live server's build_info.
DIFFERENT_BUILD_TAG = "b0000"
# A restored first request should prefill only the new tail (a short
# question); a full re-prefill of these ~150-token prompts is far beyond this.
TAIL_ONLY_MAX_TOKENS = 64
# Gap between the two saves in the LRU test, so their mtimes order
# deterministically.
LRU_SEPARATION_S = 1.1
# A disk budget that fits one saved state (plus slack), never two.
ONE_STATE_FIT = 1.5
# Bytes of KV payload stomped in the corruption test.
CORRUPTION_BYTES = 4096
# Slack for the BOS token llama-server prepends to a prompt.
BOS_SLACK = 2
# How long to wait for a compose server to finish loading the model.
READY_TIMEOUT_S = 30.0
READY_POLL_S = 0.5
_BYTES_PER_GB = 1e9

RUN_COMMAND = (
    "docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit test"
)

pytestmark = pytest.mark.skipif(
    not all((URL_PRIMARY, URL_TWIN, URL_FA_OFF, URL_SMALL_CTX, CACHE_DIR)),
    reason=(
        "integration servers not configured; run "
        f"{RUN_COMMAND} (it sets STILLWARM_TEST_URL_*/STILLWARM_TEST_CACHE_DIR)"
    ),
)

PREFIX = (
    "The five essential rules of the fictional Order of the Quiet Lantern are: "
    "First, never light two lanterns in one window. Second, always thank the "
    "lamplighter by name. Third, oil is counted on Mondays. Fourth, a broken "
    "wick must be buried at dusk. Fifth, the youngest keeper holds the key. "
)
QUESTION = "Question: What happens on Mondays?\nAnswer:"


def _fetch_props(url: str) -> dict[str, Any]:
    with urlopen(f"{url.rstrip('/')}/props", timeout=READY_TIMEOUT_S) as resp:
        props: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return props


@pytest.fixture(scope="module", autouse=True)
def wait_for_servers() -> None:
    """Compose starts the servers before this container, but model load takes
    a few seconds; wait until every server answers /health."""
    for url in (URL_PRIMARY, URL_TWIN, URL_FA_OFF, URL_SMALL_CTX):
        api = LlamaServer(url)
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not api.health():
            if time.monotonic() >= deadline:
                msg = (
                    f"llama-server at {url} not ready after {READY_TIMEOUT_S:g}s; inspect it "
                    "with: docker compose -f docker-compose.integration.yml logs"
                )
                raise TimeoutError(msg)
            time.sleep(READY_POLL_S)


@pytest.fixture(scope="module")
def primary() -> LlamaServer:
    return LlamaServer(URL_PRIMARY)


@pytest.fixture(scope="module")
def twin() -> LlamaServer:
    return LlamaServer(URL_TWIN)


@pytest.fixture(scope="module")
def fa_off() -> LlamaServer:
    return LlamaServer(URL_FA_OFF)


@pytest.fixture(scope="module")
def small_ctx() -> LlamaServer:
    return LlamaServer(URL_SMALL_CTX)


@pytest.fixture(scope="module")
def build_tag() -> str:
    """The live server's build tag (build_info from /props)."""
    props = _fetch_props(URL_PRIMARY)
    tag: str = props.get("build_info", "")
    if not tag:
        msg = f"/props from {URL_PRIMARY} has no build_info; keys: {sorted(props)}"
        raise RuntimeError(msg)
    return tag


@pytest.fixture
def cache_dir() -> Path:
    """The shared --slot-save-path directory, wiped before each test.

    The slot-save-path is fixed at server boot, so per-test isolation is a
    wipe, not a fresh directory.
    """
    d = Path(CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
    return d


def profile(fa: str = "on", ctx: int = FULL_CTX) -> ServerProfile:
    # f16/f16: the KV cache type the compose servers run with (the
    # llama-server default; no -ctk/-ctv flags are passed).
    return ServerProfile(
        model_sha256=MODEL_SHA,
        flash_attn=fa,
        cache_type_k="f16",
        cache_type_v="f16",
        swa_full=False,
        ctx_size=ctx,
    )


def make_session(
    api: LlamaServer,
    cache_dir: Path,
    build_tag: str,
    **profile_kwargs: Any,
) -> Session:
    return Session(
        api,
        profile(**profile_kwargs),
        str(cache_dir),
        server_build_tag=build_tag,
    )


def test_works_roundtrip_across_restart(
    cache_dir: Path,
    primary: LlamaServer,
    twin: LlamaServer,
    build_tag: str,
) -> None:
    """WORKS: ask -> (server dies) -> ask again; restore reuses the prefix."""
    sw = make_session(primary, cache_dir, build_tag)
    r1 = sw.ask("s1", PREFIX + QUESTION, n_predict=24, sampler=dict(PROBE_SAMPLER))
    assert not r1.restored
    assert r1.content
    # fresh process, same save volume: the saved state survives the "restart"
    sw2 = make_session(twin, cache_dir, build_tag)
    r2 = sw2.ask("s1", PREFIX + "Question: Who holds the key?\nAnswer:", n_predict=24)
    assert r2.restored
    assert r2.warning == ""  # guard silent
    # reuse evidence: only the new tail was prefilled
    assert r2.prompt_n is not None
    assert r2.prompt_n < TAIL_ONLY_MAX_TOKENS


def test_refuse_fa_mismatch(
    cache_dir: Path,
    primary: LlamaServer,
    fa_off: LlamaServer,
    build_tag: str,
) -> None:
    """REFUSE(fa): sidecar fa=on must not load into an fa=off profile."""
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX + QUESTION, n_predict=8)
    sw_off = make_session(fa_off, cache_dir, build_tag, fa="off")
    with pytest.raises(RefusedRestoreError, match="flash-attention"):
        sw_off.restore("s1")


def test_refuse_ctx_fit(
    cache_dir: Path,
    primary: LlamaServer,
    small_ctx: LlamaServer,
    build_tag: str,
) -> None:
    """REFUSE(ctx-fit): state saved at ctx 1024 must not load into ctx smaller
    than the state (sidecar n_saved_tokens > server ctx).
    """
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX * 6 + QUESTION, n_predict=8)  # ~600-token state
    n_saved = Sidecar.read(sidecar_path_for(cache_dir / "s1.bin")).n_saved_tokens
    assert n_saved > SMALL_CTX
    sw_small = make_session(small_ctx, cache_dir, build_tag, ctx=SMALL_CTX)
    with pytest.raises(RefusedRestoreError, match="does not fit"):
        sw_small.restore("s1")


def test_advisory_build_tag(cache_dir: Path, primary: LlamaServer, build_tag: str) -> None:
    """ADVISORY(build): different build tag warns but restores fine."""
    sw = make_session(primary, cache_dir, DIFFERENT_BUILD_TAG)  # saved under 'b0000'
    sw.ask("s1", PREFIX + QUESTION, n_predict=8)
    sw2 = make_session(primary, cache_dir, build_tag)  # live build tag
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sw2.restore("s1")
    assert any("build differs" in str(x.message) for x in w)


def test_verify_pass(
    cache_dir: Path,
    primary: LlamaServer,
    twin: LlamaServer,
    build_tag: str,
) -> None:
    """--verify: restored state reproduces the save-time probe hash."""
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX + QUESTION, n_predict=24)
    sw2 = make_session(twin, cache_dir, build_tag)
    assert sw2.restore("s1", verify=True) is True
    # post-verify state is the RESTORED state: next ask reuses the prefix
    r = sw2.ask("s1", PREFIX + "Question: Who is thanked?\nAnswer:", n_predict=16)
    assert r.prompt_n is not None
    assert r.prompt_n < TAIL_ONLY_MAX_TOKENS


def test_verify_mismatch_on_corrupted_copy(
    cache_dir: Path,
    primary: LlamaServer,
    twin: LlamaServer,
    build_tag: str,
) -> None:
    """--verify catches a corrupted save file (bytes flipped mid-payload).

    The corrupted copy can surface as a probe mismatch (VerifyFailedError) or
    as a server-side load failure (HTTP error or dropped connection); any
    loud failure passes, a silent one fails the test.
    """
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX + QUESTION, n_predict=24)
    # corrupt a COPY, keeping the original intact
    src = cache_dir / "s1.bin"
    dst = cache_dir / "s1corrupt.bin"
    shutil.copy(src, dst)
    for aux in ("s1.bin.stillwarm.json", "s1.prefix.txt"):
        shutil.copy(cache_dir / aux, cache_dir / aux.replace("s1", "s1corrupt", 1))
    blob = bytearray(dst.read_bytes())
    mid = len(blob) // 2
    for i in range(mid, mid + CORRUPTION_BYTES):  # stomp the KV payload
        blob[i] ^= 0xFF
    dst.write_bytes(blob)
    sw2 = make_session(twin, cache_dir, build_tag)
    with pytest.raises((VerifyFailedError, RefusedRestoreError, Exception)) as ei:
        sw2.restore("s1corrupt", verify=True)
    assert not isinstance(ei.value, AssertionError)


def test_ineffective_guard_fires_on_nonmatching_prompt(
    cache_dir: Path,
    primary: LlamaServer,
    twin: LlamaServer,
    build_tag: str,
) -> None:
    """Guard: restore then ask with a DIFFERENT prefix -> near-full re-prefill
    -> IneffectiveRestoreWarning (the guard doesn't care WHY reuse failed).
    """
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX * 3 + QUESTION, n_predict=8)
    sw2 = make_session(twin, cache_dir, build_tag)
    other = (
        "Completely different text about shipping logistics, containers and harbors. "
    ) * 12 + "Question: what is this about?\nAnswer:"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = sw2.ask("s1", other, n_predict=8)
    assert r.restored
    assert any(isinstance(x.message, IneffectiveRestoreWarning) for x in w)
    assert "SILENTLY INEFFECTIVE" in r.warning


def test_lru_prune_via_session(cache_dir: Path, primary: LlamaServer, build_tag: str) -> None:
    """Disk budget: oldest session pruned when over budget."""
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("old", PREFIX + QUESTION, n_predict=8)
    time.sleep(LRU_SEPARATION_S)
    one = (cache_dir / "old.bin").stat().st_size
    sw.disk_budget_gb = int(one * ONE_STATE_FIT) / _BYTES_PER_GB
    sw.ask("new", PREFIX + "Question: Rule one?\nAnswer:", n_predict=8)
    assert not (cache_dir / "old.bin").exists()
    assert (cache_dir / "new.bin").exists()


def test_save_point_after_generation_in_ask_flow(
    cache_dir: Path,
    primary: LlamaServer,
    build_tag: str,
) -> None:
    """Regression: ask() generates a reply, so the saved state ends
    with sampled tokens -> sidecar must say after_generation (was mislabeled).
    """
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX + QUESTION, n_predict=16, sampler=dict(PROBE_SAMPLER))
    sc = Sidecar.read(sidecar_path_for(cache_dir / "s1.bin"))
    assert sc.save_point == "after_generation"
    # numeric cross-check: state is larger than the pre-generation prompt
    pre_gen = len(primary.tokenize(PREFIX + QUESTION))
    assert sc.n_saved_tokens > pre_gen


def test_save_point_doc_prefill_in_prime_flow(
    cache_dir: Path,
    primary: LlamaServer,
    build_tag: str,
) -> None:
    """Prime flow: prefill-only state (no sampled tokens) -> doc_prefill."""
    sw = make_session(primary, cache_dir, build_tag)
    primary.slot_erase()  # fresh slot: the n_saved bound below assumes a clean start
    primary.completion(PREFIX, 0, cache_prompt=True)  # prefill, generate nothing
    sw._live_prefix["primed"] = PREFIX  # noqa: SLF001 — prime flow needs a prefill-only state
    sc = sw.save("primed")  # generated_tokens defaults to 0
    assert sc.save_point == "doc_prefill"
    assert sc.n_saved_tokens <= len(primary.tokenize(PREFIX)) + BOS_SLACK  # prompt(+BOS) only


def test_guard_message_mentions_after_generation(
    cache_dir: Path,
    primary: LlamaServer,
    twin: LlamaServer,
    build_tag: str,
) -> None:
    """Guard must use the (now-correct) save_point in its diagnosis."""
    sw = make_session(primary, cache_dir, build_tag)
    sw.ask("s1", PREFIX * 3 + QUESTION, n_predict=8)  # after_generation state
    sw2 = make_session(twin, cache_dir, build_tag)
    other = (
        "Entirely unrelated inventory report about warehouse pallets. "
    ) * 14 + "Question: topic?\nAnswer:"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r = sw2.ask("s1", other, n_predict=8)
    assert "save_point=after_generation" in r.warning
