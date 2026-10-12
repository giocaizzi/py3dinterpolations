# AGENTS.md

3D spatial interpolation library (Kriging, IDW) published to PyPI. User docs: [`docs/`](docs/index.md), built with MkDocs (`mkdocs.yml`).

## Commands

- `make help` lists targets; `make check` = `make lint` (ruff, ruff format, mypy strict) + `make test` (pytest).
- `uv` only: `uv sync --extra dev` (add `--extra docs` and `uv run mkdocs serve` for docs).

## Workflow

- Changes land on `main` only via a squash-merged PR with a Conventional Commit title; required CI job is `check` (`.github/workflows/ci.yml`). Repo conventions: [GitHub standard](https://github.com/giocaizzi/.github/blob/main/docs/standard.md).
- Versions are owned by release-please (`release-please.yml`, `release-please-config.json`, `.release-please-manifest.json`): never bump `pyproject.toml` or `__version__` by hand; the Conventional Commit PR title drives the bump.
- Publishing to PyPI runs on the GitHub release that release-please creates (`python-publish.yml`); docs deploy to GitHub Pages on push to `main` (`documentation.yml`).
