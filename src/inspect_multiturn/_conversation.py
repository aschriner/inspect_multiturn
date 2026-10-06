"""The conversation orchestrator."""

from __future__ import annotations

from typing import Literal

from inspect_ai.agent import Agent, as_solver
from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.util import LimitExceededError, span

from ._stop import StopCondition
from ._types import ConversationState, Stop, TurnInfo, UserSimulator
from .simulators import llm_user


@solver
def converse(
    target: Agent | None = None,
    user: UserSimulator | None = None,
    max_turns: int = 10,
    first_turn: Literal["dataset", "simulator"] = "dataset",
    stop_when: StopCondition | None = None,
) -> Solver:
    """Run a multi-turn conversation between a simulated user and the target.

    Each turn is one user message followed by the target's full response
    (including any tool calls it makes). By default the target is the model
    being evaluated, called through Inspect's `generate()`: tools come from
    `use_tools()` and model, config and limits from the task and eval.

    The sample's transcript is the conversation, so standard scorers and the log
    viewer work unchanged. The outcome is recorded in `ConversationState`.

    The simulator and the target are kept separate: the simulator reads the
    `TaskState` but only `converse()` writes to the transcript, and the target
    only ever sees the transcript.

    The conversation ends when the simulator returns `Stop`, after `max_turns`
    turns, when `stop_when` returns `True`, or when an Inspect limit is hit.

    Args:
        target: Optional `@agent` to run as the target instead of `generate()`,
            e.g. a custom or bridged agent. It receives the transcript and should
            return after producing one turn's response.
        user: The user simulator. Defaults to `llm_user()`.
        max_turns: Maximum number of turns, including a dataset-provided first turn.
        first_turn: `"dataset"` uses the sample input as the first user message;
            the input must end with a user message. `"simulator"` has the
            simulator write the first message; the input must not end with a user
            message (use an empty input, or only a system message).
        stop_when: Optional check run after each target turn.
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be at least 1 (got {max_turns}).")

    user = user if user is not None else llm_user()
    # Either way the target is a solver that appends one turn's response.
    target_step = as_solver(target) if target is not None else generate()

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        _check_first_turn(state, first_turn)
        conv = state.store_as(ConversationState)

        try:
            for turn in range(max_turns):
                async with span(f"turn {turn}", type="turn"):
                    if turn > 0 or first_turn == "simulator":
                        stop = await _user_turn(user, state, conv, turn)
                        if stop is not None:
                            conv.stop_reason = stop.reason
                            conv.stop_metadata = stop.metadata
                            return state

                    state = await target_step(state, generate_fn)
                    conv.turns = turn + 1

                    if stop_when is not None and await stop_when(state):
                        conv.stop_reason = "target_condition"
                        return state
        except LimitExceededError:
            conv.stop_reason = "limit"
            raise

        conv.stop_reason = "max_turns"
        return state

    return solve


async def _user_turn(
    user: UserSimulator, state: TaskState, conv: ConversationState, turn: int
) -> Stop | None:
    """Ask the simulator for its action and apply it to the transcript."""
    transcript = list(state.messages)
    async with span("user", type="user"):
        action = await user(state, TurnInfo(turn, len(conv.simulated)))
    _check_unmodified(state, transcript)

    if isinstance(action, Stop):
        return action

    # Only the content reaches the transcript. Which messages are simulated,
    # and the simulator's metadata (e.g. its reasoning), go in the store.
    message = ChatMessageUser(content=action.content)
    state.messages.append(message)
    conv.simulated[message.id] = action.metadata
    return None


def _check_unmodified(state: TaskState, transcript: list[ChatMessage]) -> None:
    if len(state.messages) != len(transcript) or any(
        a is not b for a, b in zip(state.messages, transcript, strict=True)
    ):
        raise RuntimeError(
            "The user simulator modified the transcript. Simulators must return "
            "a UserAction instead; converse() is the transcript's only writer."
        )


def _check_first_turn(
    state: TaskState, first_turn: Literal["dataset", "simulator"]
) -> None:
    ends_with_user = bool(state.messages) and isinstance(
        state.messages[-1], ChatMessageUser
    )
    if first_turn == "dataset" and not ends_with_user:
        raise ValueError(
            'first_turn="dataset" requires the sample input to end with a user '
            'message. Use first_turn="simulator" to have the simulator open.'
        )
    if first_turn == "simulator" and ends_with_user:
        raise ValueError(
            'first_turn="simulator" requires the sample input not to end with a '
            "user message (use an empty input, e.g. Sample(input=[]), or only a "
            'system message). Use first_turn="dataset" to open with the input.'
        )
