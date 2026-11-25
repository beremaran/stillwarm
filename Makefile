.PHONY: dev test test-integration download-model format lint check clean build install

dev:
	uv sync --extra dev

test:
	uv run pytest tests/

download-model:
	hf download bartowski/SmolLM2-135M-Instruct-GGUF --include "SmolLM2-135M-Instruct-Q8_0.gguf"

test-integration: download-model
	docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit test

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

check: lint format
	uv run ruff check src tests --fix
	uv run ruff format src tests
	pyrefly check

clean:
	rm -rf .venv *.egg-info build dist __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

build:
	uv build

install: dev