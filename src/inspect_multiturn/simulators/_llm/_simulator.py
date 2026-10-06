from __future__ import annotations

from typing import Any

from inspect_ai.model import (
    CachePolicy,
    ChatMessageAssistant,
    ChatMessageSystem,
    ContentReasoning,
    GenerateConfig,
    Model,
)
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field

from ..._types import Stop, TurnInfo, UserAction, UserMessage
from .._metadata import read_metadata
from .._model import resolve_user_model
from ._history import flip_roles
from ._prompt import (
    DEFAULT_PERSONA,
    DEFAULT_USER_PROMPT,
    END_CONVERSATION,
    end_conversation_tool,
)
from ._view import ViewFn, default_user_view


class LLMUserMetadata(BaseModel):
    """Sample metadata read by `llm_user()`.

    Arguments passed to `llm_user()` take precedence over these keys. Other
    metadata keys are ignored.

    Attributes:
        goal: What the user is trying to get the target to do.
        persona: Who the user is. Defaults to a generic user.
    """

    goal: str = Field(min_length=1)
    persona: str | None = None


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
            sample `metadata["goal"]`; one of the two is required. See
            `LLMUserMetadata` for the metadata schema.
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
        model = resolve_user_model(
            self._model,
            "llm_user",
            "openai/gpt-5",
            " It may be the same model as the target.",
        )
        metadata = self._resolve_metadata(state)
        system = self._prompt.format(
            persona=metadata.persona or DEFAULT_PERSONA,
            goal=metadata.goal,
            end_tool=END_CONVERSATION,
        )
        history = self._visible_to_user(state.messages)
        output = await model.generate(
            input=[ChatMessageSystem(content=system), *flip_roles(history)],
            tools=[end_conversation_tool()],
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

    def _resolve_metadata(self, state: TaskState) -> LLMUserMetadata:
        overrides = {"goal": self._goal, "persona": self._persona}
        return read_metadata(
            LLMUserMetadata,
            {**state.metadata, **{k: v for k, v in overrides.items() if v}},
            'llm_user() requires a goal: pass goal=... or set metadata["goal"].',
        )


def _reasoning_metadata(message: ChatMessageAssistant) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {}
    reasoning = [
        c.reasoning for c in message.content if isinstance(c, ContentReasoning)
    ]
    return {"reasoning": "\n\n".join(reasoning)} if reasoning else {}
