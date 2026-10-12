.PHONY: help lint test check

help: ## Show available targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "%-8s %s\n", $$1, $$2}'

lint: ## Ruff lint + format check, mypy
	uv run ruff check
	uv run ruff format --check
	uv run mypy py3dinterpolations

test: ## Run pytest
	uv run pytest

check: lint test ## lint + test
