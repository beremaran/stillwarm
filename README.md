# stillwarm

Your local LLM re-prefills everything after every restart. stillwarm fixes that.

llama.cpp's `llama-server` can save a conversation's KV cache (the model's
per-token "notes") to disk and load it back. Nothing does it automatically, and
nothing checks that the file you're loading matches the server you're loading
it into. stillwarm is a lightweight Python library (httpx + orjson + click + rich):

- restore-before-request, save-after-response — a session survives the server
  restarting;
- a metadata sidecar next to every save — an incompatible file is refused with
  a reason, never silently misloaded;
- a loud warning when a restore "succeeds" but the server re-prefills anyway
  (the measured SWA trap without `--swa-full`), so you do not pay cold-start
  cost for nothing;
- LRU pruning under a disk budget (KV files are ~128 KiB per token for an 8B
  model, so a 32K-token session is ~4.3 GB);
- `verify` — an integrity mode that byte-checks a restored state against a
  probe captured at save time.

## Install

```sh
uv add stillwarm
```

Python 3.13 or newer. Runtime dependencies: httpx, orjson.

## Quickstart

The server must be running with `--slot-save-path` set. `cache_dir` is that
directory, and stillwarm bakes in no default for it:

```python
from stillwarm import LlamaServer, ServerProfile, Session

# llama-server ... --flash-attn on --slot-save-path /path/to/cache   (must be running)
server = LlamaServer("http://127.0.0.1:8080")  # any reachable http(s) URL works, remote included
profile = ServerProfile(
    model_sha256="<sha256 of your .gguf>",
    flash_attn="on",
    cache_type_k="f16",
    cache_type_v="f16",
    swa_full=False,
    ctx_size=16384,
)
sw = Session(
    server,
    profile,
    cache_dir="/path/to/cache",
    server_build_tag="<your llama-server's build tag>",
    disk_budget_gb=50,
)

r = sw.ask("paper-review", document + "\n\nQ: Summarize.\nA:", n_predict=256)
# ... llama-server restarts ...
r = sw.ask("paper-review", document + "\n\nQ: And the weaknesses?\nA:", n_predict=256)
print(r.restored, r.prompt_n)  # True, a few dozen tokens instead of thousands
```

Inspect and prune the cache from the shell (or through the `store` module):

```
stillwarm list  /path/to/cache                           # sessions, sizes, ages
stillwarm inspect /path/to/cache/paper-review.bin        # sidecar
stillwarm prune /path/to/cache --budget-gb 50 [--dry-run]
```

Every command takes `--json` for the raw machine-readable form; human output
is a rich table by default. Exit codes: 0 success, 1 runtime error, 2 usage.

## Remote servers

`LlamaServer` accepts any reachable `http(s)` URL. One condition: your process
must be able to read the save files. They live on the server's disk (its
`--slot-save-path`), and stillwarm reads the sidecar next to each file before
restoring it. So `cache_dir` points at that directory:

- local server — `cache_dir` is the `--slot-save-path` directory;
- remote server — that directory must be mounted into your process (NFS, EFS,
  a shared volume).

If your process cannot read the save directory, there is nothing to restore,
and the `store` module cannot see the files either.

## Compatibility

Every row below is measured behavior, not speculation.

| you change... | what happens | what stillwarm does |
|---|---|---|
| flash-attention on<->off | restore fails (400) | refuses up front |
| KV cache type (f16/q8_0/q4_0) | layout mismatch | refuses up front |
| model file | meaningless state | refuses up front |
| `--swa-full` (SWA models) | restore "works" but silently re-prefills | refuses; a loud warning catches it if it ever slips through |
| server ctx < saved tokens | restore fails (fit rule) | refuses with a clear message |
| server ctx >= saved tokens | works | restores |
| `-ngl` (GPU layers) | works | restores, not part of the cache key |
| llama.cpp build (+-5 weeks measured) | works both directions | warns once, then restores |

## `verify`

At save time, stillwarm writes the slot file **first** (the file holds the
pre-probe state), then generates a 64-token pinned-greedy probe and stores its
hash in the sidecar. `restore(..., verify=True)` restores, re-runs the probe,
compares the hashes, then **re-restores** so the served state is exactly the
restored state.

This is a determinism check (does the restored state reproduce the save-time
continuation?), not a cold-equivalence check (does it match a full recompute?).
Cold-equivalence measurably fails even without corruption: different prefill
batch splits flip marginal tokens, and quantized caches amplify this to
near-certainty (q4_0: 5 of 5 divergent at 8K)
while both outputs stay coherent. A verify failure means the file, model, or
config does not reproduce what was saved. That is a real integrity signal, not
batching noise.

Cost: ~2x the restore time, plus one 64-token decode (~1.5 s for an 8B model).

## Tests

Unit tests run anywhere under `pytest`. Integration tests need real
`llama-server` processes. Docker Compose provides four of them sharing one
save volume, including a fresh-process twin that plays "the server restarted":

```sh
# one-time prerequisite: the test model (~145 MB) into your Hugging Face cache
hf download bartowski/SmolLM2-135M-Instruct-GGUF \
    --include "SmolLM2-135M-Instruct-Q8_0.gguf"

docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit test
docker compose -f docker-compose.integration.yml down -v   # wipes the save volume
```

The model is read from your Hugging Face cache
(`${HF_HOME:-$HOME/.cache/huggingface}`), mounted read-only. The image is the
rolling `ghcr.io/ggml-org/llama.cpp:server`. Without the compose environment
the integration tests skip, and the skip reason carries the run command.

## License

MIT. See [LICENSE](LICENSE).
