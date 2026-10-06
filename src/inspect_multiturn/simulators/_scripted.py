from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from inspect_ai.solver import TaskState
from pydantic import BaseModel, field_validator

from .._types import Stop, TurnInfo, UserAction, UserMessage
from ._metadata import read_metadata


class ScriptedUserMetadata(BaseModel):
    """Sample metadata read by `scripted_user()`.

    Other metadata keys are ignored.

    Attributes:
        turns: Messages to send, in order. A JSON array string is also accepted,
            so `turns` can come from a CSV column.
    """

    turns: list[str]

    @field_validator("turns", mode="before")
    @classmethod
    def _parse_json(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(
                "must be a list of strings, or a JSON array of strings"
            ) from None


def scripted_user(turns: Sequence[str] | None = None) -> _ScriptedUser:
    """A simulator that replays a fixed list of user messages.

    Deterministic and ignores the target's responses, which makes it suitable
    for regression tests and controlled comparisons between targets. Returns
    `Stop("script_exhausted")` once every message has been sent.

    Args:
        turns: Messages to send, in order. Defaults to the sample's
            `metadata["turns"]` (see `ScriptedUserMetadata`). When the
            conversation opens with the dataset input, these are the messages
            that follow it.
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
    return read_metadata(
        ScriptedUserMetadata,
        state.metadata,
        "scripted_user() without explicit turns requires sample "
        'metadata["turns"] to be a list of strings.',
    ).turns
