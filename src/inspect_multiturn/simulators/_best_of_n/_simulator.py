from __future__ import annotations

from typing import Any

from inspect_ai.solver import TaskState
from inspect_ai.util import collect

from ..._types import Stop, TurnInfo, UserAction, UserMessage, UserSimulator
from ._selector import candidate_text, llm_selector
from ._types import CandidateSelector, Selection


def best_of_n_user(
    generator: UserSimulator,
    n: int = 4,
    selector: CandidateSelector | None = None,
) -> _BestOfNUser:
    """A simulator that drafts several candidate actions and sends the best one.

    Each turn, `generator` is called `n` times concurrently, each draft
    independent of the others, and `selector` chooses one of the results. A
    natural generator is `userlm_user()`, whose drafts are realistic but
    uneven; any simulator works. Don't enable the generator's cache, or every
    draft will be the same.

    The chosen candidate is returned as-is (a `UserMessage`, or a `Stop` if the
    selector chose a draft that ends the conversation), with these keys added to
    its metadata, and so recorded in `ConversationState`:

    - `"candidates"`: every candidate, in order, as `{"content": ...,
      "metadata": ...}` or `{"stop": reason, "metadata": ...}`.
    - `"selected"`: the chosen candidate's index.
    - `"selection"`: the selector's metadata (e.g. its rationale), if any.

    Args:
        generator: Simulator that drafts the candidates, e.g.
            `userlm_user(model="vllm/microsoft/UserLM-8b")`.
        n: Number of candidates per turn. With 1, the selector isn't called.
        selector: Chooses among the candidates. Defaults to `llm_selector()`,
            which uses the `user_selector` model role. Pass a custom
            `CandidateSelector` to choose by other means, e.g. by inspecting
            the `TaskState` or the sandbox.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1 (got {n}).")
    return _BestOfNUser(generator, n, selector or llm_selector())


class _BestOfNUser:
    def __init__(
        self, generator: UserSimulator, n: int, selector: CandidateSelector
    ) -> None:
        self._generator = generator
        self._n = n
        self._selector = selector

    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction:
        candidates = await collect(
            *(self._generator(state, turn) for _ in range(self._n))
        )
        if len(candidates) == 1:
            selection = Selection(0)
        else:
            selection = await self._selector(state, turn, candidates)
        if not 0 <= selection.index < len(candidates):
            raise ValueError(
                f"The candidate selector chose index {selection.index}, but there "
                f"are only {len(candidates)} candidates."
            )

        chosen = candidates[selection.index]
        metadata: dict[str, Any] = {
            **chosen.metadata,
            "candidates": [_record(c) for c in candidates],
            "selected": selection.index,
        }
        if selection.metadata:
            metadata["selection"] = selection.metadata

        if isinstance(chosen, Stop):
            return Stop(chosen.reason, metadata)
        return UserMessage(chosen.content, metadata)


def _record(candidate: UserAction) -> dict[str, Any]:
    if isinstance(candidate, Stop):
        return {"stop": candidate.reason, "metadata": candidate.metadata}
    return {
        "content": candidate_text(candidate.content),
        "metadata": candidate.metadata,
    }
