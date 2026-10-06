from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    GenerateConfig,
    Model,
    get_model,
    model_roles,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef

from .._types import Stop, TurnInfo, UserAction, UserMessage

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

_OPENING = "[The conversation hasn't started. Write your opening message.]"
_BEGIN = "[The conversation begins.]"
_NO_REPLY = "[The assistant hasn't replied. Write your next message.]"


def llm_user(
    model: str | Model | None = None,
    persona: str | None = None,
    goal: str | None = None,
    prompt: str = DEFAULT_USER_PROMPT,
    visible_to_user: ViewFn = default_user_view,
    config: GenerateConfig | None = None,
    cache: bool | CachePolicy = False,
) -> _LLMUser:
    """A simulator that uses a model to play a goal-driven user.

    The model must be given explicitly, either as `model` or as the `user` model
    role (e.g. `--model-role user=anthropic/...`). It never falls back to the
    model being evaluated, though it may be set to the same model.

    The simulator ends the conversation by calling an `end_conversation` tool,
    producing `Stop("goal_met")` or `Stop("gave_up")` with its explanation in
    the stop metadata. Treat this as the simulator's opinion, not a success
    metric: score whether the goal was met with a separate judge.

    Args:
        model: Model for the simulator. Required unless the `user` model role
            is set.
        persona: Who the user is. Defaults to sample `metadata["persona"]`, then
            a generic user.
        goal: What the user is trying to get the target to do. Defaults to
            sample `metadata["goal"]`; one of the two is required.
        prompt: System prompt template, formatted with `{persona}`, `{goal}` and
            `{end_tool}`.
        visible_to_user: Selects which parts of the transcript the simulator's
            model is shown. The default shows only what a real user would see:
            no system prompt, tool calls, tool results or reasoning.
        config: Generation config for the simulator.
        cache: Caching behavior for simulator generations.
    """
    return _LLMUser(model, persona, goal, prompt, visible_to_user, config, cache)


class _LLMUser:
    def __init__(
        self,
        model: str | Model | None,
        persona: str | None,
        goal: str | None,
        prompt: str,
        visible_to_user: ViewFn,
        config: GenerateConfig | None,
        cache: bool | CachePolicy,
    ) -> None:
        self._model = model
        self._persona = persona
        self._goal = goal
        self._prompt = prompt
        self._visible_to_user = visible_to_user
        self._config = config or GenerateConfig()
        self._cache = cache

    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction:
        model = _resolve_model(self._model)
        system = self._prompt.format(
            persona=self._persona or state.metadata.get("persona") or DEFAULT_PERSONA,
            goal=self._resolve_goal(state),
            end_tool=END_CONVERSATION,
        )
        history = self._visible_to_user(state.messages)
        output = await model.generate(
            input=[ChatMessageSystem(content=system), *flip_roles(history)],
            tools=[_end_conversation_tool()],
            config=self._config,
            cache=self._cache,
        )

        for call in output.message.tool_calls or []:
            if call.function == END_CONVERSATION:
                goal_met = str(call.arguments.get("goal_met")).lower() == "true"
                return Stop(
                    "goal_met" if goal_met else "gave_up",
                    {"explanation": call.arguments.get("explanation", "")},
                )

        text = output.message.text.strip()
        if not text:
            return Stop("empty_response")
        return UserMessage(text, _reasoning_metadata(output.message))

    def _resolve_goal(self, state: TaskState) -> str:
        goal = self._goal or state.metadata.get("goal")
        if not goal:
            raise ValueError(
                'llm_user() requires a goal: pass goal=... or set metadata["goal"].'
            )
        return str(goal)


def _resolve_model(model: str | Model | None) -> Model:
    if model is not None:
        return get_model(model)
    if "user" not in model_roles():
        raise ValueError(
            "llm_user() needs an explicit model: pass llm_user(model=...) or set "
            "the user model role (e.g. --model-role user=openai/gpt-5). It may be "
            "the same model as the target."
        )
    return get_model(role="user")


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


def _end_conversation_tool() -> ToolDef:
    async def end_conversation(goal_met: bool, explanation: str) -> str:
        """End the conversation.

        Args:
            goal_met: Whether the assistant did what your goal describes.
            explanation: One or two sentences on why you are ending now.
        """
        return "Conversation ended."

    return ToolDef(end_conversation, name=END_CONVERSATION)


def _reasoning_metadata(message: ChatMessageAssistant) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {}
    reasoning = [
        c.reasoning for c in message.content if isinstance(c, ContentReasoning)
    ]
    return {"reasoning": "\n\n".join(reasoning)} if reasoning else {}
