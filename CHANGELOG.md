# Changelog

## Unreleased

- Add `converse()`, a solver that runs a multi-turn conversation between a
  simulated user and the target, recording the outcome (including which messages
  were simulated) in `ConversationState`. The target is the evaluated model via
  `generate()` (tools from `use_tools()`), or any Inspect agent passed as
  `target=`. Usable from the CLI as `--solver inspect_multiturn/converse`.
- Add user simulators `llm_user()`, `scripted_user()` and `fn_user()`. Simulators
  receive the sample's `TaskState` and a `TurnInfo`. `llm_user()` requires its
  model to be set explicitly (`model=` or the `user` model role).
- Add `tool_called()` for ending a conversation when the target calls a given tool.
