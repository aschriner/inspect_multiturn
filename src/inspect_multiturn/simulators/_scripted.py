from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.solver import TaskState

from .._types import Stop, TurnInfo, UserAction, UserMessage


def scripted_user(turns: Sequence[str] | None = None) -> _ScriptedUser:
    """A simulator that replays a fixed list of user messages.

    Deterministic and ignores the target's responses, which makes it suitable
    for regression tests and controlled comparisons between targets. Returns
    `Stop("script_exhausted")` once every message has been sent.

    Args:
        turns: Messages to send, in order. Defaults to the sample's
            `metadata["turns"]`. When the conversation opens with the dataset
            input, these are the messages that follow it.
    """
    return _ScriptedUser(turns)


class _ScriptedUser:
    def __init__(self, turns: Sequence[str] | None) -> None:
        self._turns = list(turns) if turns is not None else None

    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction:
        turns = self._turns if self._turns is not None else _metadata_turns(state)
        if turn.user_turn >= len(turns):
            return Stop("script_exhausted")
        return UserMessage(turns[turn.user_turn])


def _metadata_turns(state: TaskState) -> list[str]:
    turns = state.metadata.get("turns")
    if not isinstance(turns, list) or not all(isinstance(t, str) for t in turns):
        raise ValueError(
            "scripted_user() without explicit turns requires sample "
            'metadata["turns"] to be a list of strings.'
        )
    return turns
