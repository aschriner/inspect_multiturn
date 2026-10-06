"""Target-side conditions that end a conversation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import TaskState

StopCondition = Callable[[TaskState], Awaitable[bool]]
"""Checked after each target turn; returning `True` ends the conversation."""


def tool_called(*names: str) -> StopCondition:
    """Stop when the target calls any of the named tools during a turn.

    Args:
        *names: Tool names, e.g. `"transfer_to_human"`.
    """
    if not names:
        raise ValueError("tool_called() requires at least one tool name.")

    async def check(state: TaskState) -> bool:
        for message in reversed(state.messages):
            if isinstance(message, ChatMessageUser):
                return False
            if isinstance(message, ChatMessageAssistant) and any(
                call.function in names for call in message.tool_calls or []
            ):
                return True
        return False

    return check
