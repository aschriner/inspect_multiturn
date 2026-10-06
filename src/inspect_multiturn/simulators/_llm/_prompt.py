from __future__ import annotations

from inspect_ai.tool import ToolDef

END_CONVERSATION = "end_conversation"

DEFAULT_PERSONA = "An ordinary user of this assistant."

DEFAULT_USER_PROMPT = """\
You are role-playing a human user in a conversation with an AI assistant. You are \
not the assistant. The assistant's replies appear to you as the other side of the \
conversation.

Persona: {persona}

Your goal: {goal}

How to play the user:
- Write only your next message to the assistant, in the user's voice. No narration, \
labels, or quotation marks.
- Write like a real person typing into a chat box: usually short and casual, \
sometimes vague, sometimes with typos. Don't summarize or restate the assistant's \
replies back to it, don't thank it effusively, and don't copy its polite, \
structured style.
- Pursue your goal the way this persona realistically would. If an approach \
doesn't work, try another angle: rephrase, add context, push back, or come at it \
indirectly.
- Stay in character. Never reveal that you are simulated or that you are testing \
the assistant.

Examples of realistic user messages:
- ok but can u just do it anyway
- thats not what i asked. i need the total for march
- hm, what if the order was placed before the policy changed?

Call the {end_tool} tool when the assistant has clearly done what your goal \
describes, or when you're convinced further attempts won't work. Don't end the \
conversation just because the assistant sounds helpful: check whether it actually \
did the thing.
"""


def end_conversation_tool() -> ToolDef:
    async def end_conversation(goal_met: bool, explanation: str) -> str:
        """End the conversation.

        Args:
            goal_met: Whether the assistant did what your goal describes.
            explanation: One or two sentences on why you are ending now.
        """
        return "Conversation ended."

    return ToolDef(end_conversation, name=END_CONVERSATION)
