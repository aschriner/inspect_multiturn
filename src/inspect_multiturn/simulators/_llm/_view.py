from __future__ import annotations

from collections.abc import Callable

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser

ViewFn = Callable[[list[ChatMessage]], list[ChatMessage]]
"""Selects the parts of the transcript that `llm_user` shows its model."""


def default_user_view(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Keep only what a real user sees: user messages and assistant text.

    System messages, tool results, tool calls and reasoning are removed.
    Assistant messages with no visible text (e.g. tool-call-only steps) are dropped.

    Args:
        messages: The target's transcript.

    Returns:
        New messages; the input is not modified.
    """
    view: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, ChatMessageUser):
            view.append(message.model_copy())
        elif isinstance(message, ChatMessageAssistant):
            text = message.text.strip()
            if text:
                view.append(ChatMessageAssistant(content=text, model=message.model))
    return view
