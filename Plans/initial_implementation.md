# inspect_multiturn: design and roadmap

**Status (2026-10-06):** Phase 1 is implemented, plus `fn_user`. Phases 2 and 3 are
not started. This document describes the current design. Where it differs from the
original plan, the [decision log](#decision-log) at the end records what changed and
why.

**Primary use case:** elicitation. A simulated user with a persona and a goal holds
a multi-turn conversation with the target, to see whether persistent pressure
elicits a behavior that a single prompt doesn't.

## Where it lives: an extension

This is a standalone package, `inspect_multiturn`, that depends on `inspect_ai`. It
is not a change to Inspect core. The reasons:

- **Core already has the primitives.** Solvers, agents, the Store, spans, limits and
  model roles are all public API. The implementation uses no private Inspect APIs.
- **User simulation is opinionated and changes fast.** Persona prompting, stop
  semantics, realism filtering and rollback are research questions, not settled
  infrastructure. Core Inspect has a high bar for API stability and ships on AISI's
  release cadence (AISI is the UK AI Security Institute, which maintains Inspect).
  This API is expected to break several times in its first few months.
- **The ecosystem already works this way.** The closest prior art is Petri, an
  extension maintained by Meridian Labs as `inspect_petri`. It orchestrates
  multi-turn audits with an auditor model and a target model. It's a collaboration
  between Meridian Labs and the UK AISI Red Team, based on work originally done by
  the Alignment team at Anthropic, so even AISI-adjacent multi-turn work lives
  outside core.

The strongest argument for core is discoverability, plus one canonical
"conversation" primitive instead of many incompatible ones. The answer is to keep
the protocol small and free of dependencies. If it stabilizes, the protocol and
the orchestrator could be proposed for upstreaming, while the simulators stay in
the extension.

**Scope.** Petri covers adversarial auditing with rollback and synthetic tools. This
package is the general-purpose layer: goal-driven users, scripted turns and replay.
It should not become a second Petri. A Petri-style auditor can later be one
pluggable simulator among several.

## Core design decisions

**1. A turn is one user message plus the target's full response,** including any
tool calls the target makes along the way.

**2. The target is the evaluated model, configured the usual Inspect way.** By
default, each target turn is Inspect's `generate()` solver, which produces one reply
and resolves tool calls in a loop. Tools come from `use_tools(...)`. Model,
generation config and limits come from the task and eval. To evaluate something
other than a plain model, pass any `@agent` as `converse(target=...)`; it runs
through Inspect's `as_solver`. The library ships no target code of its own.

If you pass `react()` as a target, be aware it's built for completing tasks: it
loops until `submit` and injects "continue" messages, which doesn't fit
conversational turns.

**3. The orchestrator, `converse()`, is a solver.** It receives the full
`TaskState` and Inspect's `generate`, so everything it needs (metadata, sample id,
epoch, the store) comes through public API. It's the only thing that writes to the
transcript. The transcript is the sample's ordinary message history, so standard
scorers and the log viewer work unchanged. Each turn and each simulator call is
wrapped in a log span. With no arguments, `converse()` also works from the command
line as `--solver inspect_multiturn/converse`.

**4. The simulator is entirely separate from the target, but not restricted.**
- **Simulators may read the whole `TaskState`:** transcript, metadata, sample id,
  epoch and store. A simulator may legitimately need any of these.
- **They may not modify the transcript.** `converse()` raises an error if one does.
- **Nothing a simulator produces reaches the target except the message content.**
  Simulated messages enter the transcript as plain user messages with no marker.
  Which messages were simulated, and the simulator's private metadata (such as its
  reasoning), are recorded only in the Store.
- **What a simulator shows its own model is that simulator's decision.**
  `llm_user` shows only what a real user would see by default, using
  `default_user_view`: no system prompt, tool calls, tool results or reasoning. Pass
  `visible_to_user=...` to change it. `llm_user` also swaps the roles in that
  history for its own prompt.

**5. The simulator returns an action, not a string.** `UserAction = UserMessage |
Stop`. `Rollback` is deferred to Phase 3. Adding it then won't break existing
simulators, which simply never return it.

**6. Simulators are stateless.** One simulator instance is shared by every sample in
a task, and samples run concurrently. A simulator's position comes from the
`TurnInfo` it's passed (`turn`, and `user_turn` = messages it has sent so far),
never from instance attributes.

**7. The simulator's model must be chosen explicitly.** `llm_user` needs either
`model=` or the `user` model role (`--model-role user=...`). It may be the same
model as the target, but it never silently defaults to it.

**8. Termination is layered.** A conversation ends when:
- the simulator returns `Stop` (`stop_reason` is the `Stop.reason`),
- `max_turns` is reached (`"max_turns"`; the max includes a dataset-provided first
  turn),
- the optional `stop_when` check fires after a target turn (`"target_condition"`,
  for example via `tool_called("transfer_to_human")`), or
- an Inspect limit is hit (`"limit"`; the error is re-raised so Inspect handles it
  as usual).

Token and time limits should be scoped to the target, so the simulator's usage
doesn't eat the target's budget and skew comparisons between simulators. This is
not built yet (Phase 2). Note that wrapping each target turn in `token_limit(...)`
won't work, because the budget would reset every turn. It has to be cumulative
across turns.

**9. The package runs conversations; it doesn't score them.** The outcome goes in a
`ConversationState` store model for scorers to read. A simulator's own claim that
it met its goal is recorded but is not a score. Reference scorers are Phase 2. For
now, `examples/refund_policy.py` includes a simple judge.

## API (as built)

```python
# _types.py
@dataclass
class UserMessage:
    content: str | list[Content]
    metadata: dict[str, Any] = field(default_factory=dict)  # private to the simulator


@dataclass
class Stop:
    reason: str  # "goal_met", "gave_up", "script_exhausted", ...
    metadata: dict[str, Any] = field(default_factory=dict)


UserAction = UserMessage | Stop


@dataclass(frozen=True)
class TurnInfo:
    turn: int  # 0-based turn being produced; a dataset-provided opening is turn 0
    user_turn: int  # messages this simulator has produced so far


class UserSimulator(Protocol):
    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction: ...


class ConversationState(StoreModel):
    turns: int = 0
    stop_reason: str | None = None
    stop_metadata: dict[str, Any] = Field(default_factory=dict)
    simulated: dict[str, dict[str, Any]] = Field(default_factory=dict)  # message id -> metadata
```

```python
# _conversation.py
@solver
def converse(
    target: Agent | None = None,  # None: the evaluated model via generate()
    user: UserSimulator | None = None,  # None: llm_user()
    max_turns: int = 10,
    first_turn: Literal["dataset", "simulator"] = "dataset",
    stop_when: StopCondition | None = None,  # async (TaskState) -> bool
) -> Solver: ...


# In a task:
#   solver=[system_message(...), use_tools(...), converse(user=llm_user(), max_turns=8)]
```

`first_turn` is checked before the conversation starts. `"dataset"` requires the
sample input to end with a user message. `"simulator"` requires that it doesn't:
use `Sample(input=[])` or a system message only.

## Simulators

**`scripted_user(turns=None)`** (built) replays fixed messages, from the argument or
`metadata["turns"]`, and returns `Stop("script_exhausted")` at the end. It's
deterministic and ignores the target's responses, which makes it good for
regression tests and for controlled comparisons between versions of a target.

**`llm_user(model=None, persona=None, goal=None, ...)`** (built) is the workhorse. A
model plays a user pursuing `goal` as `persona`, which come from the arguments or
from sample metadata. It ends the conversation by calling an `end_conversation`
tool, producing `Stop("goal_met")` or `Stop("gave_up")` with the model's
explanation. An empty reply becomes `Stop("empty_response")`. Its prompt template
(`DEFAULT_USER_PROMPT`) is written for elicitation and includes measures against two
predictable failure modes:
- **Voice drift.** Simulators drift into an assistant's voice: over-polite, verbose,
  summarizing the target back to itself. The prompt forbids this explicitly and
  gives examples of terse, realistic user messages. A realism check is a later
  option.
- **Premature satisfaction.** Instruction-tuned models are biased toward declaring
  the goal met. The prompt tells the simulator to check whether the target actually
  did the thing, and the simulator's verdict is never used as the score.

**`fn_user(fn)`** (built) is a plain-function escape hatch for rule-based branching.
The function may be sync or async, and may return a `str` as shorthand for a
`UserMessage`.

**`replay_user(log_path)`** (Phase 2) will take user turns from an existing eval
log, matching samples by sample id and epoch, both of which are available on the
`TaskState`. This enables "N+1" evaluation: fix the first N turns and measure only
the target's behavior at turn N+1. That's the closest multi-turn evaluation gets to
a controlled experiment, because the random simulator is removed from everything
except the turn under test.

**`human_user()`** (Phase 3) will prompt at the terminal, for debugging and
baselines. Inspect's human-agent tooling may be reusable here.

**Auditor adapter** (Phase 3). Once `Rollback` exists, a Petri-style controller can
be wrapped as a simulator.

## Measurement and reproducibility

With an LLM simulator, an eval has two sources of randomness: the target and the
user. Plan for that explicitly:
- Use `epochs` generously, and report the variance per sample, not just means.
- Pin the simulator's model and temperature, and log both. The model is already
  explicit (decision 7), and `llm_user(config=...)` sets the temperature.
- Cache the simulator's generations (`llm_user(cache=...)`, which uses Inspect's
  cache) so that rerunning an unchanged target reproduces the same users. The cache
  key includes the simulator's prompt, and therefore the visible history. Once
  rollback exists, branches with identical input must not share a cache entry.
  Petri handles this by scoping target generations by trajectory path.
- When comparing versions of a target, prefer scripted or replay users where
  possible.

## Package layout

```
src/inspect_multiturn/
  _types.py          # UserMessage, Stop, UserAction, TurnInfo, UserSimulator, ConversationState
  _conversation.py   # converse()
  _stop.py           # StopCondition, tool_called()
  _registry.py       # inspect_ai entry point; registers converse
  simulators/        # _llm.py (llm_user, default_user_view), _scripted.py, _fn.py
examples/            # refund_policy.py: elicitation task with a judge scorer
tests/               # offline; mockllm only
```

Planned: `simulators/_replay.py` and `simulators/_human.py`, and a `scorers/`
package.

## Phasing

- **Phase 1 (done):**
  - the protocol types, `converse()`, `tool_called()`
  - `scripted_user`, `llm_user` and `fn_user`
  - the `refund_policy` elicitation example
  - an offline test suite, and CI on Python 3.10–3.13
- **Phase 2:**
  - `replay_user`
  - reference scorers (whole-conversation judge, per-turn judge)
  - cumulative token and time limits scoped to the target
- **Phase 3:** `Rollback`, `human_user`, and an auditor adapter.

The check the original plan asked for before Phase 1 is done: Inspect 0.3.268 has
no built-in user simulation.

## Open questions

- **Scoring shape for elicitation.** The example judge gives a yes/no verdict. Real
  evals may need a graded scale, the first turn at which the behavior appeared, or
  both. This decides what the Phase 2 reference scorers look like.
- **Parameter order.** The target is usually left at its default, so most calls
  read `converse(user=...)`. Swapping to `converse(user, target=None, ...)` would be
  a cheap change before the first release.
- **The example judge's model.** It uses the `grader` role and falls back to the
  evaluated model. It isn't covered by the separation rule; decide whether it
  should require an explicit model like `llm_user` does.

## Decision log

Changes from the original plan, in the order they were made:

1. **Primary use case: elicitation.** This answered the original plan's open
   question. `llm_user`'s prompt and the example are written for it. `Rollback`
   stayed out of v1.
2. **`llm_user` ends conversations with a tool only.** The original plan mentioned
   structured output (`reasoning`, `message`, `done`) as an alternative. The tool
   was chosen because it works with any provider that supports tool calling.
3. **`ConversationView` gained `user_turn`, and `ConversationState` gained
   `stop_metadata`.** Simulators are shared across concurrent samples, so they need
   their position handed to them, and `Stop.metadata` would otherwise have been lost.
4. **`max_turns` counts the dataset turn, and `first_turn` is checked up front.** The
   original sketch never counted the dataset turn in `turns`. The upfront check
   prevents two user messages in a row.
5. **`target_token_limit` was deferred to Phase 2.** A per-turn `token_limit` resets
   every turn, so a correct version needs a cumulative budget.
6. **`conversation()` was renamed `converse()`.** A verb says it runs something, and
   it's no longer confusable with the `Conversation*` types.
7. **The orchestrator became a solver, not an agent.** As an agent it could only get
   sample metadata through Inspect's private `sample_state()`. Phase 2's
   `replay_user` would also have needed the sample id and epoch, which agents don't
   receive. The main benefit of an agent, nesting, didn't hold up either: two nested
   conversations would have overwritten each other's `ConversationState`.
8. **Simulators get the full `TaskState`, not a filtered view.** The rule that
   matters is separation from the target, not limits on what the simulator can read.
   `ConversationView` became `TurnInfo`. `visible_to_user` moved from the
   orchestrator into `llm_user`, as that simulator's own policy. `converse()`
   enforces that it's the only thing writing to the transcript.
9. **Simulator metadata was removed from transcript messages.** A Python target
   agent could have read the simulator's reasoning from message metadata.
10. **The `{"simulated": True}` marker was removed.** It would have let a target tell
    it was being tested. Simulated messages are identified through
    `ConversationState` instead.
11. **`llm_user` requires an explicit model.** It no longer falls back to the
    evaluated model. It may still be set to the same model.
12. **`chat_target()` was removed.** Once `converse()` was a solver, Inspect's
    `generate()` did the same job in the standard Inspect way (`use_tools`, task
    config, built-in limits). Custom targets go through `as_solver`.
13. **More simplifications.** The target runner was removed: both kinds of target
    now go through one solver. `simulated_message_ids` and `user_turn_metadata` were
    merged into one `simulated` dict. `Rollback` was removed until Phase 3. The view
    function moved into `simulators/_llm.py`. The simulator half of each turn moved
    into a helper.
