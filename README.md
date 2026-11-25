# stillwarm

Your local LLM re-prefills everything after every restart. stillwarm fixes that.

llama.cpp's `llama-server` can save a conversation's KV cache (the model's
per-token "notes") to disk and load it back — but nothing does it automatically,
and nothing checks that the file you're loading matches the server you're
loading it into. stillwarm is a zero-dependency Python library + small CLI:

- restore-before-request, save-after-response — sessions survive restarts;
- a metadata sidecar per save — incompatible files are refused with a reason, never silently misloaded;
- a guard against silent failure — if a restore "succeeds" but the server
  re-prefills anyway (the measured SWA-without-`--swa-full` trap), you get a
  loud warning instead of paying cold-start cost for nothing;
- LRU pruning under a disk budget (KV files are ~128 KiB/token for an 8B
  model — a 32K-token session is ~4.3 GB);
- `verify` — an integrity mode that byte-checks a restored state against a
  probe captured at save time.

Measured on an M3 Max (llama.cpp b9871, Llama-3.1-8B Q4_K_M, same-session
thermally controlled): resuming a session from disk instead of re-prefilling it
is **71x faster at 8K tokens, 159x at 32K, 201x at 64K** (restore + first-token
time vs cold prefill), and ~50x cheaper in energy at 32K. Even with a purged
page cache, restore wins for any document longer than ~300 tokens.

## Quickstart

```python
from stillwarm import LlamaServer, ServerProfile, Session

# llama-server ... -fa on --slot-save-path ~/.stillwarm/   (must be running)
server = LlamaServer("http://127.0.0.1:8080")
profile = ServerProfile(
    model_sha256="<sha256 of your .gguf>",
    flash_attn="on",
    cache_type_k="f16",
    cache_type_v="f16",
    swa_full=False,
    ctx_size=16384,
)
sw = Session(server, profile, cache_dir="~/.stillwarm", server_build_tag="b9871", disk_budget_gb=50)

r = sw.ask("paper-review", document + "\n\nQ: Summarize.\nA:", n_predict=256)
# ... llama-server restarts ...
r = sw.ask("paper-review", document + "\n\nQ: And the weaknesses?\nA:", n_predict=256)
print(r.restored, r.prompt_n)  # True, ~a few dozen tokens instead of thousands
```

`cache_dir` must be the directory passed to `--slot-save-path`. CLI:

```
stillwarm list  ~/.stillwarm            # sessions, sizes, ages
stillwarm inspect ~/.stillwarm/paper-review.bin
stillwarm prune ~/.stillwarm --budget-gb 50 [--dry-run]
```

## Compatibility contract

Every rule below was measured, not guessed.

From a 480-row benchmark matrix ([stillwarm-bench], Block D portability taxonomy):

| you change... | what happens | stillwarm's response |
|---|---|---|
| flash-attention on<->off | restore fails (400) | **REFUSE** up front |
| KV cache type (f16/q8_0/q4_0) | layout mismatch | **REFUSE** |
| model file | meaningless state | **REFUSE** |
| `--swa-full` (SWA models) | restore "works" but silently re-prefills | **REFUSE** + runtime guard |
| server ctx < saved tokens | restore fails (fit rule) | **REFUSE** with a clear message |
| server ctx >= saved tokens | works | OK |
| `-ngl` (GPU layers) | works | OK — not part of the cache key |
| llama.cpp build (+-5 weeks measured) | works both directions | **ADVISORY** warning only |

## `verify`

At save time, stillwarm writes the slot file **first** (the file is the
pre-probe state), then generates a 64-token pinned-greedy probe and stores its
hash in the sidecar. `restore(..., verify=True)` restores, re-runs the probe,
compares hashes, then **re-restores** so the served state is exactly the
restored state.

This is a determinism check (does the restored state reproduce the
save-time continuation?), not a cold-equivalence check (does it match a full
recompute?). Cold-equivalence measurably fails even without corruption:
different prefill batch splits flip marginal tokens, and quantized caches
amplify this to near-certainty (q4_0: 5/5 divergent at 8K in
[stillwarm-bench] Block C) while both outputs stay coherent. A verify failure
means the file/model/config does not reproduce what was saved — a real
integrity signal, not batching noise.

Cost: ~2x restore time + one 64-token decode (~1.5 s for an 8B model).

## Non-goals

- **Not a serving framework** — one machine, one llama-server.
- **No multi-machine cache sharing**; no vLLM (LMCache owns that space).
- **llama-server only**: Ollama and LM Studio do not expose the slot
  save/restore endpoints this needs.
- No compression research.

Small and finished beats large and abandoned.

## Supported platforms

- macOS (Apple Silicon) — tested; all numbers come from an M3 Max running
  llama.cpp b9871.
- Linux — should work (same llama-server HTTP API; the demo Space runs on
  Linux/CPU). Not benchmarked.
- Windows — untested. The code avoids POSIX-only modules and uses atomic
  writes via `os.replace`; reports welcome.

## Tests

`pytest` — unit tests run anywhere; integration tests need a `llama-server`
binary and a tiny GGUF (defaults: SmolLM2-135M-Instruct Q8_0, ~145 MB; override
via `STILLWARM_TEST_SERVER_BIN` / `STILLWARM_TEST_MODEL`; they skip if absent).

## License

MIT — see [LICENSE](LICENSE).

[stillwarm-bench]: https://github.com/vimalnakrani08/stillwarm-bench
