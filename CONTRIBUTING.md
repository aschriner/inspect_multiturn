# Contributing

Thanks for your interest in contributing! For larger changes, please open an issue to
discuss the idea before sending a pull request.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/). With uv installed:

```bash
git clone https://github.com/aschriner/inspect_multiturn.git
cd inspect_multiturn
uv sync
```

This creates `.venv/` with the package (editable) and all dev tools. Either activate
it (`source .venv/bin/activate`) or prefix commands with `uv run`.

## Checks

```bash
uv run ruff check --fix   # lint
uv run ruff format        # format
uv run pytest             # test
```

CI runs these on every pull request, across Python 3.10–3.13.

## Pull requests

- Keep changes focused, and include tests for new behavior.
- Add an entry under `## Unreleased` in [CHANGELOG.md](CHANGELOG.md) for any
  user-visible change.
