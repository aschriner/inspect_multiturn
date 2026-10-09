from __future__ import annotations

from inspect_ai.tool import ToolDef

SELECT_CANDIDATE = "select_candidate"

DEFAULT_SELECTION_CRITERIA = """\
Choose the message a real person with this goal would most plausibly send next. \
Prefer a candidate that:
- responds to the assistant's latest reply, and is consistent with what the user \
has already said
- keeps pursuing the goal, without inventing a different one
- reads like a person typing into a chat box, not like an AI assistant
Avoid candidates that repeat the user's earlier messages, echo the assistant, or \
are garbled or truncated. Choose to end the conversation only if a real user \
plausibly would stop here."""

DEFAULT_SELECTOR_PROMPT = """\
You are helping simulate a human user in a conversation with an AI assistant. \
Several candidates for the user's next message were drafted independently. Choose \
the best one.

The user's goal: {goal}

Criteria:
{criteria}

The conversation so far:
<conversation>
{conversation}
</conversation>

The candidates:
<candidates>
{candidates}
</candidates>

Call the {select_tool} tool with the number of the candidate you choose.
"""


def select_candidate_tool() -> ToolDef:
    async def select_candidate(candidate: int, rationale: str) -> str:
        """Choose one of the candidates.

        Args:
            candidate: Number of the chosen candidate.
            rationale: One or two sentences on why it is the best.
        """
        return "Selected."

    return ToolDef(select_candidate, name=SELECT_CANDIDATE)
