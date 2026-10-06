# Repo guidance for coding agents

`inspect_multiturn` is an open source [Inspect](https://inspect.aisi.org.uk/)
extension for multi-turn evaluations. It is published to PyPI as
`inspect-multiturn` and imported as `inspect_multiturn`.

## Tooling decisions

These are settled; don't introduce alternatives (pip/poetry, black/flake8/isort,
unittest/nose) without discussion.

- **Package management: [uv](https://docs.astral.sh/uv/).** Dependencies live in
  `pyproject.toml`; dev-only tools go in `[dependency-groups] dev`. Add them with
  `uv add <pkg>` / `uv add --dev <pkg>`, never by hand-editing and forgetting the
  lockfile. `uv.lock` is committed and CI runs `uv sync --locked`, so update it in
  the same change as `pyproject.toml`.
- **Linting and formatting: [ruff](https://docs.astral.sh/ruff/).** Configuration is in
  `pyproject.toml` (`[tool.ruff]`). Docstrings follow the Google convention.
- **Testing: [pytest](https://docs.pytest.org/).** Tests live in `tests/`.
  `pytest-asyncio` runs in auto mode, so `async def test_...` works without a marker.

## Commands

```bash
uv sync                      # create .venv and install the package + dev tools
uv run ruff check --fix      # lint (and autofix)
uv run ruff format           # format
uv run pytest                # run the test suite
```

Before considering a change done, all of `ruff check`, `ruff format --check`, and
`pytest` must pass. CI runs the same checks on Python 3.10–3.13.

## Layout

```
src/inspect_multiturn/
  __init__.py        # public API: re-export everything users should import
  _registry.py       # imported by Inspect via the `inspect_ai` entry point
  _types.py          # protocol: UserAction, TurnInfo, UserSimulator, ConversationState
  _conversation.py   # converse(), the orchestrator solver
  _stop.py           # target-side stop conditions
  simulators/        # UserSimulator implementations
  py.typed           # PEP 561 marker: the package ships type hints
examples/            # runnable example tasks (smoke-tested in tests/)
Plans/               # design notes; Plans/initial_implementation.md is the roadmap
tests/
```

## Design invariants

- `converse()` is a solver, and the only writer of the transcript
  (`state.messages`). The target is the evaluated model via `generate()`, or an
  agent the developer passes in; either way it sees only the transcript. The
  library ships no target implementations of its own.
- The simulator is kept entirely separate from the target: it never shares a model
  call or context with the target, and nothing it produces besides the
  `UserMessage` content reaches the target. Simulators may *read* the whole
  `TaskState`; `converse()` raises if one modifies the transcript.
- What a simulator shows its own model is that simulator's policy (e.g.
  `llm_user(visible_to_user=...)`), not something the orchestrator restricts.
- Simulators are shared across concurrent samples, so they must be stateless.
  Derive position from `TurnInfo.user_turn`, not from instance attributes.
- The package runs conversations; it does not decide success. A simulator's
  `Stop` reason is recorded but is not a score.

## Conventions

- **Registering components.** Anything decorated with `@task`, `@solver`, `@scorer`,
  `@tool`, `@modelapi`, `@sandboxenv`, etc. must be imported in `_registry.py` so
  Inspect can resolve it by name (e.g. `--solver inspect_multiturn/<name>`). Public
  components should also be re-exported from `__init__.py` and listed in `__all__`.
- **Private modules.** Implementation modules are underscore-prefixed
  (`_something.py`); the public surface is what `__init__.py` exports.
- **Python support.** `requires-python = ">=3.10"`, matching Inspect. Don't use
  syntax or stdlib APIs newer than 3.10.
- **Type hints** on all public functions; docstrings on all public objects.
- **Tests must not call real model APIs.** Use Inspect's `mockllm/model` (and
  `ModelOutput.from_content(...)` for canned responses) so the suite runs offline
  and for free. Mark genuinely slow tests with `@pytest.mark.slow`.

## Changelog (`CHANGELOG.md`)

User-visible changes go under `## Unreleased`. Write entries for someone *using* the
extension: describe the observable change (new names, breaking-change migration
steps, the symptom a bug fix removes), not the implementation.
