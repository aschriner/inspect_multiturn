from __future__ import annotations

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser

_OPENING = "[The conversation hasn't started. Write your opening message.]"
_BEGIN = "[The conversation begins.]"
_NO_REPLY = "[The assistant hasn't replied. Write your next message.]"


def flip_roles(history: list[ChatMessage]) -> list[ChatMessage]:
    """Rewrite the target's history from the simulated user's perspective.

    The target's user messages become the simulator's own (assistant) turns, and
    the target's replies become incoming (user) turns. The result always starts
    and ends with a user message, as most providers require.

    Args:
        history: History with roles from the target's perspective, already
            filtered by `visible_to_user`.
    """
    flipped: list[ChatMessage] = []
    for message in history:
        if isinstance(message, ChatMessageUser):
            flipped.append(ChatMessageAssistant(content=message.content))
        elif isinstance(message, ChatMessageAssistant):
            flipped.append(ChatMessageUser(content=message.content))

    if not flipped:
        return [ChatMessageUser(content=_OPENING)]
    if isinstance(flipped[0], ChatMessageAssistant):
        flipped.insert(0, ChatMessageUser(content=_BEGIN))
    if isinstance(flipped[-1], ChatMessageAssistant):
        flipped.append(ChatMessageUser(content=_NO_REPLY))
    return flipped
