# Agent Directives for stillwarm

## What this is

Python 3.13+ library (src layout, single package) that auto-saves and restores
llama-server KV-cache sessions, with a JSON sidecar that lets it refuse
incompatible files. Module map:

- `server.py` — httpx client; the only module that talks HTTP
- `session.py` — the core loop: restore -> complete -> save
- `sidecar.py` — the `<id>.bin.stillwarm.json` metadata next to each save
- `compat.py` — pure REFUSE/ADVISORY/OK decision logic
- `store.py` — list/prune sessions under a disk budget
- `settings.py` — `STILLWARM_*` env vars via pydantic-settings

## Workflow

- uv is the environment manager; drive everything through the Makefile
  targets (`make dev|test|lint|format|check|build`) or `uv run`.
- Trunk-based development: commit and push directly to `main`. No branches,
  no PRs, and there is no CI — the Makefile and pre-commit are the gates.
- Focused verification: `uv run pytest tests/test_store.py -k name`
  (single file / single test, standard pytest).

## Tooling quirks

- ruff runs `select = ["ALL"]` (line-length 100): pydocstyle D rules apply to
  every public def in src, plus security and many more rule families. Only
  `tests/` gets per-file relaxations.
- pyrefly (preset "strict", migrated from mypy strict) is the only type
  checker. The codebase is fully annotated and `# type: ignore` is a hard
  error (`unused-type-ignore`).

## Invariants (read before changing)

- Save order in `Session.save()`: `slot_save` runs BEFORE probe generation,
  so the .bin holds the pre-probe state. `verify` is a determinism check
  against that state; reordering the two breaks it.
- A stored session is a triple: `<id>.bin` + `<id>.bin.stillwarm.json` +
  `<id>.prefix.txt` (the prefix text is what `--verify` probes). Store
  operations must treat the triple as one unit.
- `cache_dir` must equal the server's `--slot-save-path` directory: stillwarm
  reads the sidecars from disk, so the process must be able to read that
  directory even for remote servers (via a mount).

## Docs stance

Every compatibility claim in the README and docstrings is MEASURED llama.cpp
behavior, not speculation (see the README compatibility table). Keep that bar:
when adding a behavior claim, state what was measured; never write a guess as
a fact.

## Tests

- `make test` runs the unit suite anywhere (no Docker). `test_integration.py`
  self-skips when the `STILLWARM_TEST_*` env vars are absent; the skip message
  carries the run command.
- Real integration run: `make test-integration` — one-time model download into
  the host's Hugging Face cache (~145 MB), then four llama-server containers
  under Docker Compose sharing one save volume (the "twin" container is the
  fresh process that plays "the server restarted"). Tear down with
  `docker compose -f docker-compose.integration.yml down -v` (wipes the volume).

## Versioning

`pyproject.toml` is the source of truth for the version; keep
`stillwarm/__init__.py: __version__` in sync with it.
