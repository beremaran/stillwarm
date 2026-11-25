"""Integration tests against real llama-server processes (Docker Compose).

Covers the measured Block-D taxonomy end-to-end: WORKS, REFUSE(fa),
REFUSE(ctx-fit), ADVISORY(build), verify-pass, verify-mismatch (corrupted
copy), and the SILENTLY-INEFFECTIVE guard path. Plus the `stillwarm proxy`
middleware: cold first turn, multi-turn restore with tail-only prefill,
chat-probe reproduction across a fresh process, refused restores (fa and
ctx-fit), byte-identical passthrough, and header vs derived session ids.

The servers and proxies are provided by docker-compose.integration.yml:
four llama-server containers and four stillwarm-proxy containers (one per
upstream, the proxy's design contract) sharing one save volume. The "twin"
is a fresh process that plays "the server restarted". The test container
talks to everything over HTTP only and never touches the docker daemon.

Environment (all required; the module skips with the run command if any is
missing):
  STILLWARM_TEST_URL_PRIMARY    ctx 1024, flash-attn on
  STILLWARM_TEST_URL_TWIN       ctx 1024, flash-attn on, fresh process
  STILLWARM_TEST_URL_FA_OFF     ctx 1024, flash-attn off
  STILLWARM_TEST_URL_SMALL_CTX  ctx 256, flash-attn on
  STILLWARM_TEST_PROXY_PRIMARY  stillwarm proxy in front of the primary
  STILLWARM_TEST_PROXY_TWIN     stillwarm proxy in front of the twin
  STILLWARM_TEST_PROXY_FA_OFF   stillwarm proxy in front of fa-off
  STILLWARM_TEST_PROXY_SMALL_CTX stillwarm proxy in front of small-ctx
  STILLWARM_TEST_CACHE_DIR      directory shared with the servers'
                                --slot-save-path and the proxies'
                                --cache-dir

Run:
  docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit test
"""

import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from stillwarm import (
    PROBE_SAMPLER,
    IneffectiveRestoreWarning,
    LlamaServer,
    RefusedRestoreError,
    ServerProfile,
    Session,
    VerifyFailedError,
)
from stillwarm.proxy import derived_session_id
from stillwarm.sidecar import Sidecar, sidecar_path_for


class _IntegrationSettings(BaseSettings):
    """Integration test configuration from environment."""

    model_config = SettingsConfigDict(
        env_prefix="STILLWARM_TEST_",
        env_file=".env",
        extra="ignore",
    )

    url_primary: str = ""
    url_twin: str = ""
    url_fa_off: str = ""
    url_small_ctx: str = ""
    proxy_primary: str = ""
    proxy_twin: str = ""
    proxy_fa_off: str = ""
    proxy_small_ctx: str = ""
    cache_dir: str = ""


ts = _IntegrationSettings()
URL_PRIMARY = ts.url_primary
URL_TWIN = ts.url_twin
URL_FA_OFF = ts.url_fa_off
URL_SMALL_CTX = ts.url_small_ctx
PROXY_PRIMARY = ts.proxy_primary
PROXY_TWIN = ts.proxy_twin
PROXY_FA_OFF = ts.proxy_fa_off
PROXY_SMALL_CTX = ts.proxy_small_ctx
CACHE_DIR = ts.cache_dir

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
# Per-chat-request client timeout for the proxy tests (the 135M model is fast;
# the probe adds a 64-token decode per save).
CHAT_TIMEOUT_S = 60.0
# Fraction of the saved tokens the server must report as cached after a
# restored turn, to count as a real reuse (a full re-prefill reports ~0).
RESTORE_REUSE_FRACTION = 0.75
# How long to wait for a proxy's post-response save (slot_save, then the
# 64-token chat probe, then the triple write) to land in the shared volume.
SAVE_TIMEOUT_S = 60.0

RUN_COMMAND = (
    "docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit test"
)

pytestmark = pytest.mark.skipif(
    not all(
        (
            URL_PRIMARY,
            URL_TWIN,
            URL_FA_OFF,
            URL_SMALL_CTX,
            PROXY_PRIMARY,
            PROXY_TWIN,
            PROXY_FA_OFF,
            PROXY_SMALL_CTX,
            CACHE_DIR,
        )
    ),
    reason=(
        "integration servers not configured; run "
        f"{RUN_COMMAND} (it sets STILLWARM_TEST_URL_*/STILLWARM_TEST_PROXY_*/"
        "STILLWARM_TEST_CACHE_DIR)"
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
    resp = httpx.get(f"{url.rstrip('/')}/props", timeout=READY_TIMEOUT_S)
    return resp.json()  # type: ignore[no-any-return]


@pytest.fixture(scope="module", autouse=True)
def wait_for_servers() -> None:
    """Compose starts the servers before this container, but model load takes
    a few seconds; wait until every server answers /health. The proxies wait
    for their upstream's /props before exec'ing (see the compose file), so
    /health through a proxy only answers once BOTH are up — wait for those
    too."""
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
    for url in (PROXY_PRIMARY, PROXY_TWIN, PROXY_FA_OFF, PROXY_SMALL_CTX):
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            try:
                resp = httpx.get(f"{url.rstrip('/')}/health", timeout=READY_POLL_S)
                ready = resp.status_code == 200
            except httpx.RequestError:
                ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                msg = (
                    f"stillwarm proxy at {url} not ready after {READY_TIMEOUT_S:g}s; inspect it "
                    "with: docker compose -f docker-compose.integration.yml logs proxy-primary"
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


# --- The proxy middleware (stillwarm proxy) ----------------------------------
#
# One proxy per upstream (compose), all reading the shared save volume. The
# per-test cache_dir wipe is the same isolation the library tests use: the
# proxies hold nothing in-process across requests. The bad-id test runs
# first: it asserts an empty volume, and every chat test ends with an
# in-flight save (the proxy saves after the response).


def _chat_body(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A pinned-sampler chat body: short, deterministic, non-streaming."""
    body: dict[str, Any] = {
        "model": "smollm",
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0,
        "top_k": 1,
    }
    body.update(extra)
    return body


def _proxy_chat(url: str, body: dict[str, Any], *, session_id: str | None = None) -> httpx.Response:
    """POST a chat request through a proxy; optional explicit session header."""
    headers = {"X-Stillwarm-Session": session_id} if session_id else {}
    url_full = f"{url}/v1/chat/completions"
    return httpx.post(url_full, json=body, headers=headers, timeout=CHAT_TIMEOUT_S)


def _sidecar_mtime_ns(base: Path, session_id: str) -> int | None:
    """The session's sidecar mtime, or None if the triple is not there yet."""
    sc_path = sidecar_path_for(base / f"{session_id}.bin")
    try:
        return sc_path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _wait_for_triple(
    base: Path,
    session_id: str,
    *,
    newer_than_ns: int | None = None,
    timeout: float = SAVE_TIMEOUT_S,
) -> Sidecar:
    """The proxy saves AFTER the response (slot_save -> chat probe -> triple
    write), so the triple is not guaranteed to have landed when the response
    returns. Wait for it (or, with newer_than_ns, for a rewrite of it) and
    return the sidecar."""
    bin_path = base / f"{session_id}.bin"
    sc_path = sidecar_path_for(bin_path)
    prefix_path = base / f"{session_id}.prefix.txt"
    deadline = time.monotonic() + timeout
    while True:
        try:
            mtime_ns = sc_path.stat().st_mtime_ns
            landed = bin_path.exists() and prefix_path.exists() and mtime_ns > (newer_than_ns or 0)
        except FileNotFoundError:
            landed = False  # a failed save rolls the triple back: keep waiting
        if landed:
            return Sidecar.read(sc_path)
        if time.monotonic() >= deadline:
            msg = (
                f"triple for session {session_id!r} did not land within {timeout:g}s; "
                "inspect it with: docker compose -f docker-compose.integration.yml logs"
            )
            raise TimeoutError(msg)
        time.sleep(READY_POLL_S)


SYSTEM = "You are a lamp keeper of the Quiet Order. Answer in one short sentence."


def test_proxy_rejects_bad_session_ids(cache_dir: Path) -> None:
    """Session ids become filenames: anything outside [A-Za-z0-9._-]{1,64} is
    a 400 from the proxy, with no upstream traffic."""
    body = _chat_body(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PREFIX + QUESTION},
        ]
    )
    for bad in ("bad id", "x" * 65, "../../evil"):
        resp = httpx.post(
            f"{PROXY_PRIMARY}/v1/chat/completions",
            json=body,
            headers={"X-Stillwarm-Session": bad},
            timeout=CHAT_TIMEOUT_S,
        )
        assert resp.status_code == 400, bad
        assert "X-Stillwarm-Restore" not in resp.headers
    assert not any(cache_dir.iterdir())


def test_proxy_first_turn_cold_and_probed_triple(cache_dir: Path) -> None:
    """First chat through the proxy: forwarded cold, then a probed triple lands
    in the shared volume (probe ON by default)."""
    body = _chat_body(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PREFIX + QUESTION},
        ]
    )
    resp = _proxy_chat(PROXY_PRIMARY, body, session_id="px-cold")
    assert resp.status_code == 200
    assert resp.headers["X-Stillwarm-Restore"] == "cold"
    data = resp.json()
    assert data["choices"][0]["message"]["content"]
    assert data["usage"]["completion_tokens"] > 0

    # The save (with its probe) completes after the response: wait for it.
    sc = _wait_for_triple(cache_dir, "px-cold")
    assert sc.probe_kind == "chat"
    assert sc.probe_hash  # the pinned chat probe ran against the real model
    assert sc.n_saved_tokens > 0
    assert (cache_dir / "px-cold.prefix.txt").read_bytes()  # the replayable probe record


def test_proxy_second_turn_restores_with_tail_only_prefill(cache_dir: Path) -> None:
    """Multi-turn: the client resends its full history (assistant reply
    included); the proxy restores and the server prefills only the new tail."""
    user1 = PREFIX + QUESTION
    r1 = _proxy_chat(
        PROXY_PRIMARY,
        _chat_body(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user1},
            ]
        ),
        session_id="px-turn",
    )
    assert r1.status_code == 200
    reply1 = r1.json()["choices"][0]["message"]["content"]
    assert reply1
    n_saved = _wait_for_triple(cache_dir, "px-turn").n_saved_tokens  # turn-1 state

    r2 = _proxy_chat(
        PROXY_PRIMARY,
        _chat_body(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": "Who holds the key?"},
            ]
        ),
        session_id="px-turn",
    )
    assert r2.status_code == 200
    assert r2.headers["X-Stillwarm-Restore"] == "ok"
    data = r2.json()
    # reuse evidence: most of the saved state came from the slot, not a re-prefill
    cached = data["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached >= RESTORE_REUSE_FRACTION * n_saved
    assert data["timings"]["prompt_n"] < TAIL_ONLY_MAX_TOKENS


def test_proxy_cross_restart_restore_and_chat_probe_reproduction(
    cache_dir: Path, twin: LlamaServer, build_tag: str
) -> None:
    """A session saved through proxy-primary restores through proxy-twin (the
    fresh upstream), and its chat probe reproduces the save-time hash when
    verified on the twin (verify replays .prefix.txt via /v1/chat/completions)."""
    user1 = PREFIX + QUESTION
    r1 = _proxy_chat(
        PROXY_PRIMARY,
        _chat_body(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user1},
            ]
        ),
        session_id="px-verify",
    )
    assert r1.status_code == 200
    reply1 = r1.json()["choices"][0]["message"]["content"]
    sc1 = _wait_for_triple(cache_dir, "px-verify")  # the turn-1 state on disk
    before = _sidecar_mtime_ns(cache_dir, "px-verify")

    # The twin proxy restores the same triple into its own (fresh) upstream.
    r2 = _proxy_chat(
        PROXY_TWIN,
        _chat_body(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": "Who holds the key?"},
            ]
        ),
        session_id="px-verify",
    )
    assert r2.status_code == 200
    assert r2.headers["X-Stillwarm-Restore"] == "ok"
    sc2 = _wait_for_triple(cache_dir, "px-verify", newer_than_ns=before)
    assert sc2.n_saved_tokens > sc1.n_saved_tokens  # the turn-2 state replaced it
    cached = r2.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
    # r2 restored the turn-1 state: reuse evidence against THAT sidecar
    assert cached >= RESTORE_REUSE_FRACTION * sc1.n_saved_tokens

    # Chat-probe reproduction on the fresh process: verify replays the pinned
    # chat request and hashes the message exactly as the save did.
    sw = make_session(twin, cache_dir, build_tag)
    assert sw.restore("px-verify", verify=True) is True


def test_proxy_restore_refused_on_fa_off_upstream(cache_dir: Path) -> None:
    """REFUSE(fa) through the middleware: the fa-off proxy refuses the fa-on
    triple, erases its slot, forwards cold, and the client still gets its
    completion (with the refused marker)."""
    body = _chat_body(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PREFIX + QUESTION},
        ]
    )
    assert _proxy_chat(PROXY_PRIMARY, body, session_id="px-fa").status_code == 200
    _wait_for_triple(cache_dir, "px-fa")  # the fa-off refusal needs the triple

    resp = _proxy_chat(PROXY_FA_OFF, body, session_id="px-fa")
    assert resp.status_code == 200
    assert resp.headers["X-Stillwarm-Restore"] == "refused"
    assert resp.json()["choices"][0]["message"]["content"]
    # refusal erases the upstream slot, never the saved triple
    assert (cache_dir / "px-fa.bin").exists()


def test_proxy_restore_refused_when_state_exceeds_ctx(cache_dir: Path) -> None:
    """REFUSE(ctx-fit) through the middleware: a state saved at ctx 1024 must
    not restore into the ctx-256 upstream; the oversized cold forward then
    fails at the server, and the proxy relays that error."""
    big = PREFIX * 6 + QUESTION
    body = _chat_body(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": big},
        ]
    )
    assert _proxy_chat(PROXY_PRIMARY, body, session_id="px-ctx").status_code == 200
    sc = _wait_for_triple(cache_dir, "px-ctx")
    assert sc.n_saved_tokens > SMALL_CTX
    before = _sidecar_mtime_ns(cache_dir, "px-ctx")

    resp = _proxy_chat(PROXY_SMALL_CTX, body, session_id="px-ctx")
    assert resp.headers["X-Stillwarm-Restore"] == "refused"
    # the upstream rejects the oversized prompt (measured b9871: 400 + text)
    assert resp.status_code == 400
    assert "exceeds the available context size" in resp.text
    # the failed cold forward saved nothing: the primary's triple is intact
    assert _sidecar_mtime_ns(cache_dir, "px-ctx") == before


def test_proxy_passthrough_is_byte_identical() -> None:
    """Every non-chat path is relayed verbatim — GET, non-GET methods, and
    slot-cache ops alike — with no stillwarm header added."""
    # /slots/0?action=erase reports how many tokens it removed, so it is not
    # idempotent: quiesce the shared slot (erase until a no-op) so the
    # proxy-issued and direct-issued requests below see identical state.
    deadline = time.monotonic() + 10.0
    erased: int | None = None
    while time.monotonic() < deadline:
        erased = httpx.post(
            f"{URL_PRIMARY}/slots/0?action=erase", content=b"{}", timeout=CHAT_TIMEOUT_S
        ).json()["n_erased"]
        if erased == 0:
            break
        time.sleep(0.2)
    assert erased == 0, "the slot never went quiet; a save is still in flight"

    cases = (
        ("GET", "/props", None),
        ("GET", "/health", None),
        # slot-cache op: serialized under the proxy's lock, still verbatim
        ("POST", "/slots/0?action=erase", b"{}"),
        # non-GET method on an unknown path: the upstream's 404, unchanged
        ("PUT", "/nope", b"{}"),
    )
    for method, path, body in cases:
        via_proxy = httpx.request(
            method, f"{PROXY_PRIMARY}{path}", content=body, timeout=CHAT_TIMEOUT_S
        )
        direct = httpx.request(method, f"{URL_PRIMARY}{path}", content=body, timeout=CHAT_TIMEOUT_S)
        assert via_proxy.status_code == direct.status_code, path
        assert via_proxy.content == direct.content, path
        assert "X-Stillwarm-Restore" not in via_proxy.headers, path


def test_proxy_chat_path_passthrough_for_non_chat_body() -> None:
    """POST /v1/chat/completions without a messages list is not a chat request
    the proxy can reason about: it passes through, unannotated."""
    body = {"foo": "bar"}
    via_proxy = httpx.post(
        f"{PROXY_PRIMARY}/v1/chat/completions", json=body, timeout=CHAT_TIMEOUT_S
    )
    direct = httpx.post(f"{URL_PRIMARY}/v1/chat/completions", json=body, timeout=CHAT_TIMEOUT_S)
    assert via_proxy.status_code == direct.status_code
    assert via_proxy.content == direct.content
    assert "X-Stillwarm-Restore" not in via_proxy.headers


def test_proxy_header_id_vs_derived_id(cache_dir: Path) -> None:
    """The X-Stillwarm-Session header names the file; without it, the id is
    derived from [system, first user] — the same conversation lives in two
    independent triples, and each restores on its own."""
    body = _chat_body(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PREFIX + QUESTION},
        ]
    )
    assert _proxy_chat(PROXY_PRIMARY, body, session_id="px-hdr").status_code == 200
    assert _proxy_chat(PROXY_PRIMARY, body).status_code == 200

    derived = derived_session_id(body)
    assert derived != "px-hdr"
    _wait_for_triple(cache_dir, "px-hdr")
    _wait_for_triple(cache_dir, derived)
    assert (cache_dir / "px-hdr.bin").exists()
    assert (cache_dir / f"{derived}.bin").exists()
    # the second derived turn restores from its own triple
    resp = _proxy_chat(PROXY_PRIMARY, body)
    assert resp.headers["X-Stillwarm-Restore"] == "ok"
