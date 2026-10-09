from __future__ import annotations

from typing import Any

from inspect_ai.model import (
    CachePolicy,
    ChatMessageSystem,
    GenerateConfig,
    Model,
)
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field

from ..._types import Stop, TurnInfo, UserAction, UserMessage
from .._llm import ViewFn, default_user_view
from .._metadata import read_metadata
from .._model import resolve_user_model
from ._parse import normalize, parse_draft
from ._template import request

# Sampling used for UserLM in the model card and paper.
_DEFAULT_CONFIG = GenerateConfig(temperature=1.0, top_p=0.8)


class UserLMMetadata(BaseModel):
    """Sample metadata read by `userlm_user()`.

    Arguments passed to `userlm_user()` take precedence over these keys. Other
    metadata keys are ignored.

    Attributes:
        goal: The user's intent: what they want from the conversation.
    """

    goal: str = Field(min_length=1)


def userlm_user(
    model: str | Model | None = None,
    goal: str | None = None,
    prompt: str = "{goal}",
    visible_to_user: ViewFn = default_user_view,
    min_words: int | None = 3,
    max_words: int | None = None,
    max_retries: int = 5,
    config: GenerateConfig | None = None,
    cache: bool | CachePolicy = False,
) -> _UserLMUser:
    """A simulator backed by a user language model such as UserLM-8b.

    [UserLM-8b](https://huggingface.co/microsoft/UserLM-8b) (Naous et al.,
    "Flipping the Dialogue", ICLR 2026) is trained to write the *user* side of a
    conversation from a short intent. It gets no role-play instructions and no
    role flipping: its prompt is the intent as a system message followed by the
    conversation as-is, and it writes the next user turn.

    That only works if the prompt is rendered with UserLM's own chat template,
    which ends by opening a *user* turn:

    - With a completions provider (any provider whose name ends in
      `-completions`, e.g. `openai-api-completions/...` or
      `vllm-completions/...`), the simulator renders the template itself and
      sends the raw prompt. This is the reliable choice for hosted endpoints.
    - `vllm/...`, `sglang/...` and `hf/...` load the template from the model
      repo, so their chat APIs work too.
    - Other chat endpoints typically apply a generic Llama 3 template that opens
      an *assistant* turn, so UserLM writes the target's side of the
      conversation. The simulator logs a warning when used this way.

    The model ends the conversation by emitting `<|endconversation|>`, which
    becomes `Stop("end_conversation")`. Most providers strip special tokens, so
    an empty response is treated the same way.

    Some providers don't stop at the end of UserLM's turn, so it goes on to
    write the assistant's. The draft is cut at the turn boundary, which shows
    up even with special tokens stripped (e.g. "can u just tell meassistant
    Sure!"). A draft that starts with the assistant's turn (e.g. "assistantSure,
    here's how...") means the user's turn was empty, which also ends the
    conversation.

    Drafts that fail the paper's guardrails are regenerated, up to `max_retries`
    times, after which the simulator returns `Stop("guardrails_exhausted")`. A
    draft is rejected if it is too short or too long, repeats one of the user's
    earlier messages, or echoes one of the target's messages or the intent
    (UserLM occasionally parrots its context instead of replying).
    Rejected drafts are recorded in the message's `"rejected"` metadata. The
    paper's token-level guardrails (blocking the first-token openers "I", "You"
    and "Here", or the end-of-conversation token) depend on provider logit-bias
    support; pass them in `config` if your provider honours it.

    Args:
        model: Model for the simulator, e.g. `vllm/microsoft/UserLM-8b`.
            Required unless the `user` model role is set.
        goal: The user's intent, e.g. "You are a user who wants a refund for an
            order placed 45 days ago." Defaults to sample `metadata["goal"]`;
            one of the two is required. The paper uses high-level intents that
            leave details for the model to fill in.
        prompt: System message template, formatted with `{goal}`.
        visible_to_user: Selects which parts of the transcript the model is
            shown. The default shows only what a real user would see.
        min_words: Regenerate drafts with fewer words. `None` disables.
        max_words: Regenerate drafts with more words. `None` disables. The
            paper used 25 for its simulations.
        max_retries: Regenerations allowed per turn after the first draft.
        config: Generation config, merged over the defaults (`temperature=1.0`,
            `top_p=0.8`, and `max_tokens=512` for completions providers).
        cache: Caching behavior for the first draft of each turn. Regenerations
            are never cached, so they can differ.
    """
    return _UserLMUser(
        model=model,
        goal=goal,
        prompt=prompt,
        visible_to_user=visible_to_user,
        min_words=min_words,
        max_words=max_words,
        max_retries=max_retries,
        config=config,
        cache=cache,
    )


class _UserLMUser:
    def __init__(
        self,
        *,
        model: str | Model | None,
        goal: str | None,
        prompt: str,
        visible_to_user: ViewFn,
        min_words: int | None,
        max_words: int | None,
        max_retries: int,
        config: GenerateConfig | None,
        cache: bool | CachePolicy,
    ) -> None:
        self._model = model
        self._goal = goal
        self._prompt = prompt
        self._visible_to_user = visible_to_user
        self._min_words = min_words
        self._max_words = max_words
        self._max_retries = max_retries
        self._config = _DEFAULT_CONFIG.merge(config) if config else _DEFAULT_CONFIG
        self._cache = cache

    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction:
        model = resolve_user_model(
            self._model, "userlm_user", "vllm/microsoft/UserLM-8b"
        )
        system = self._prompt.format(goal=self._resolve_metadata(state).goal)
        history = self._visible_to_user(state.messages)
        messages = [ChatMessageSystem(content=system), *history]
        input, config = request(model, messages, self._config)

        # Verbatim repeats to reject: the user's own earlier messages, and
        # anything else in the prompt, which UserLM sometimes parrots back.
        own = {normalize(m.text) for m in history if m.role == "user"}
        echoes = {normalize(m.text) for m in messages if m.role != "user"}

        rejected: list[dict[str, str]] = []
        for attempt in range(self._max_retries + 1):
            output = await model.generate(
                input=input,
                config=config,
                cache=self._cache if attempt == 0 else False,
            )
            text, ended = parse_draft(output.message.text)
            if ended:
                return Stop("end_conversation", {"text": text} if text else {})

            reason = self._rejection(text, own, echoes)
            if reason is None:
                return UserMessage(text, {"rejected": rejected} if rejected else {})
            rejected.append({"text": text, "reason": reason})

        return Stop("guardrails_exhausted", {"rejected": rejected})

    def _rejection(self, text: str, own: set[str], echoes: set[str]) -> str | None:
        words = len(text.split())
        if self._min_words is not None and words < self._min_words:
            return "too_short"
        if self._max_words is not None and words > self._max_words:
            return "too_long"
        normalized = normalize(text)
        if normalized in own:
            return "repeated"
        if normalized in echoes:
            return "echoed"
        return None

    def _resolve_metadata(self, state: TaskState) -> UserLMMetadata:
        metadata: dict[str, Any] = dict(state.metadata)
        if self._goal:
            metadata["goal"] = self._goal
        return read_metadata(
            UserLMMetadata,
            metadata,
            'userlm_user() requires a goal: pass goal=... or set metadata["goal"].',
        )
