from __future__ import annotations

import logging

from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageUser,
    Content,
    ContentText,
    GenerateConfig,
    Model,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolFunction
from pydantic import BaseModel

from ..._types import Stop, TurnInfo, UserAction
from .._llm import ViewFn, default_user_view
from .._metadata import read_metadata
from .._model import resolve_user_model
from ._prompt import (
    DEFAULT_SELECTION_CRITERIA,
    DEFAULT_SELECTOR_PROMPT,
    SELECT_CANDIDATE,
    select_candidate_tool,
)
from ._types import Selection

logger = logging.getLogger(__name__)

_NO_CONVERSATION = "[The conversation hasn't started; the candidates open it.]"
_NO_GOAL = "(not specified)"


class LLMSelectorMetadata(BaseModel):
    """Sample metadata read by `llm_selector()`.

    Arguments passed to `llm_selector()` take precedence over these keys. Other
    metadata keys are ignored.

    Attributes:
        goal: The simulated user's goal, shown to the selector as context.
        selection_criteria: What makes a candidate the best one.
    """

    goal: str | None = None
    selection_criteria: str | None = None


def llm_selector(
    model: str | Model | None = None,
    criteria: str | None = None,
    goal: str | None = None,
    prompt: str = DEFAULT_SELECTOR_PROMPT,
    visible_to_user: ViewFn = default_user_view,
    config: GenerateConfig | None = None,
    cache: bool | CachePolicy = False,
) -> _LLMSelector:
    """A candidate selector that asks a model to choose the best candidate.

    The model is shown the user's goal, the selection criteria, the
    conversation and the numbered candidates, and must call a
    `select_candidate` tool with its choice and a rationale. The rationale is
    recorded in the chosen message's metadata as `selection["rationale"]`.
    Candidates that end the conversation are shown as such, so the model can
    choose to end it.

    If the model doesn't make a valid choice, the first candidate is used and
    the problem is recorded as `selection["error"]`.

    Args:
        model: Model for the selector. Required unless the `user_selector`
            model role is set.
        criteria: What makes a candidate the best one. Defaults to sample
            `metadata["selection_criteria"]`, then
            `DEFAULT_SELECTION_CRITERIA` (realism and consistency with the goal
            and conversation).
        goal: The simulated user's goal. Defaults to sample `metadata["goal"]`;
            optional.
        prompt: Prompt template, formatted with `{goal}`, `{criteria}`,
            `{conversation}`, `{candidates}` and `{select_tool}`.
        visible_to_user: Selects which parts of the transcript the selector's
            model is shown. The default shows only what a real user would see.
        config: Generation config for the selector.
        cache: Caching behavior for selector generations.
    """
    return _LLMSelector(model, criteria, goal, prompt, visible_to_user, config, cache)


class _LLMSelector:
    def __init__(
        self,
        model: str | Model | None,
        criteria: str | None,
        goal: str | None,
        prompt: str,
        visible_to_user: ViewFn,
        config: GenerateConfig | None,
        cache: bool | CachePolicy,
    ) -> None:
        self._model = model
        self._criteria = criteria
        self._goal = goal
        self._prompt = prompt
        self._visible_to_user = visible_to_user
        self._config = config or GenerateConfig()
        self._cache = cache

    async def __call__(
        self, state: TaskState, turn: TurnInfo, candidates: list[UserAction]
    ) -> Selection:
        model = resolve_user_model(
            self._model, "llm_selector", "openai/gpt-5", role="user_selector"
        )
        metadata = self._resolve_metadata(state)
        content = self._prompt.format(
            goal=metadata.goal or _NO_GOAL,
            criteria=metadata.selection_criteria or DEFAULT_SELECTION_CRITERIA,
            conversation=_render_conversation(self._visible_to_user(state.messages)),
            candidates=_render_candidates(candidates),
            select_tool=SELECT_CANDIDATE,
        )
        output = await model.generate(
            input=[ChatMessageUser(content=content)],
            tools=[select_candidate_tool()],
            tool_choice=ToolFunction(SELECT_CANDIDATE),
            config=self._config,
            cache=self._cache,
        )

        for call in output.message.tool_calls or []:
            if call.function != SELECT_CANDIDATE:
                continue
            number = _as_int(call.arguments.get("candidate"))
            if number is None or not 1 <= number <= len(candidates):
                return _fallback(
                    f"invalid candidate {call.arguments.get('candidate')!r}"
                )
            return Selection(
                number - 1, {"rationale": call.arguments.get("rationale", "")}
            )
        return _fallback(f"no {SELECT_CANDIDATE} call")

    def _resolve_metadata(self, state: TaskState) -> LLMSelectorMetadata:
        overrides = {"goal": self._goal, "selection_criteria": self._criteria}
        return read_metadata(
            LLMSelectorMetadata,
            {**state.metadata, **{k: v for k, v in overrides.items() if v}},
            "Check the llm_selector() goal and selection_criteria metadata.",
        )


def candidate_text(content: str | list[Content]) -> str:
    """The text of a message's content, ignoring non-text parts."""
    if isinstance(content, str):
        return content
    return "\n".join(c.text for c in content if isinstance(c, ContentText))


def _render_conversation(history: list[ChatMessage]) -> str:
    if not history:
        return _NO_CONVERSATION
    return "\n\n".join(f"{m.role.upper()}: {m.text}" for m in history)


def _render_candidates(candidates: list[UserAction]) -> str:
    return "\n".join(
        f'<candidate number="{number}">{_describe(candidate)}</candidate>'
        for number, candidate in enumerate(candidates, start=1)
    )


def _describe(candidate: UserAction) -> str:
    if isinstance(candidate, Stop):
        return f"[Ends the conversation without sending a message: {candidate.reason}]"
    return candidate_text(candidate.content)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _fallback(error: str) -> Selection:
    logger.warning(f"llm_selector() made no valid choice ({error}); using candidate 1.")
    return Selection(0, {"error": error})
