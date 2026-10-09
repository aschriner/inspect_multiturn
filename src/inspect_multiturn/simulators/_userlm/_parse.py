from __future__ import annotations

import re

END_CONVERSATION_TOKEN = "<|endconversation|>"

# Tokens that end a turn. Providers usually stop on them, but if one comes back,
# everything after it is another turn.
_TURN_END_TOKENS = ("<|eot_id|>", "<|end_of_text|>")

# A turn header from UserLM's template, which starts the next turn.
_HEADER = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>")

# The same header after the provider strips its special tokens: the bare role name
# glued to the text before it, e.g. "can u just tell meassistant\nSure!".
_STRIPPED_HEADER = re.compile(r"(?<=\S)(?:system|user|assistant)\n")

# A draft that is nothing but the start of another role's turn (e.g. "assistantSure,
# here's how..."): the model wrote an empty user turn, and the provider kept going
# past the end of it. Like an empty response, this ends the conversation.
_OTHER_ROLE_OPENING = re.compile(r"(?:system|assistant)(?:\n|[A-Z])")


def parse_draft(text: str) -> tuple[str, bool]:
    """Split a draft into the user's turn and what it does.

    Returns the turn's text and whether the model ended the conversation.
    """
    text = _HEADER.sub(r"\1\n", text)
    for token in _TURN_END_TOKENS:
        text = text.split(token)[0]

    text = text.strip()
    while text.startswith("user\n"):
        text = text.removeprefix("user\n").strip()
    if _OTHER_ROLE_OPENING.match(text):
        return "", True
    if header := _STRIPPED_HEADER.search(text):
        text = text[: header.start()]

    ended = END_CONVERSATION_TOKEN in text or not text.strip()
    return text.split(END_CONVERSATION_TOKEN)[0].strip(), ended


def normalize(text: str) -> str:
    return " ".join(text.lower().split())
