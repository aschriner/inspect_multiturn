# Changelog

## Unreleased

- Add `converse()`, a solver that runs a multi-turn conversation between a
  simulated user and the target, recording the outcome (including which messages
  were simulated) in `ConversationState`. The target is the evaluated model via
  `generate()` (tools from `use_tools()`), or any Inspect agent passed as
  `target=`. Usable from the CLI as `--solver inspect_multiturn/converse`.
  The simulator is the first parameter (`converse(llm_user(), max_turns=8)`); all
  other parameters are keyword-only.
- `converse(first_turn=...)` defaults to `"auto"`: the sample input opens the
  conversation if it ends with a user message, and the simulator opens it
  otherwise. Pass `"dataset"` or `"simulator"` to require one or the other.
- Add user simulators `llm_user()`, `scripted_user()` and `fn_user()`. Simulators
  receive the sample's `TaskState` and a `TurnInfo`. `llm_user()` requires its
  model to be set explicitly (`model=` or the `user` model role).
- Add `LLMUserMetadata` and `ScriptedUserMetadata`, the sample-metadata schemas
  read by `llm_user()` and `scripted_user()`. Invalid metadata raises an error
  naming the offending key (e.g. `metadata["turns"][1]`), and `turns` may be a
  JSON array string, so scripted turns can come from a CSV column.
- Add `userlm_user()`, a simulator for user language models such as
  [UserLM-8b](https://huggingface.co/microsoft/UserLM-8b) (e.g.
  `userlm_user(model="vllm/microsoft/UserLM-8b")`). It gives the model the goal as
  its intent and the conversation unflipped, ends with `Stop("end_conversation")`
  when the model emits `<|endconversation|>`, and regenerates drafts that are too
  short, too long or repeat an earlier user message (`min_words`, `max_words`,
  `max_retries`). Its metadata schema is `UserLMMetadata` (`goal`).
  With a completions provider (`openai-api-completions/...`,
  `vllm-completions/...`) it renders UserLM's chat template itself, for hosted
  endpoints whose chat API would otherwise make UserLM write the assistant's turn.
- Add `tool_called()` for ending a conversation when the target calls a given tool.
