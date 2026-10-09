from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from inspect_ai.solver import TaskState

from ..._types import TurnInfo, UserAction


@dataclass
class Selection:
    """A selector's choice among candidate user actions.

    Attributes:
        index: 0-based index of the chosen candidate.
        metadata: Selector-private data about the choice (e.g. its rationale).
            Recorded as the `"selection"` key of the chosen action's metadata.
    """

    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateSelector(Protocol):
    """Chooses one of several candidate actions for `best_of_n_user()`.

    The default is `llm_selector()`, which asks a model to judge the candidates
    against some criteria. A custom selector can use anything available to a
    solver: the whole `TaskState` (messages, metadata, store, ...), its own model
    calls, or tools in the sample's sandbox (`inspect_ai.util.sandbox()`), e.g.
    to check which candidate is consistent with the environment's state. Like
    simulators, selectors must not modify the transcript, and are shared across
    concurrent samples, so they must be stateless.
    """

    async def __call__(
        self, state: TaskState, turn: TurnInfo, candidates: list[UserAction]
    ) -> Selection:
        """Return the chosen candidate.

        Args:
            state: The sample's current state. Treat as read-only.
            turn: Position in the conversation.
            candidates: Independently generated candidates, at least two. A
                candidate may be a `Stop`, if the generator chose to end the
                conversation in that draft.
        """
        ...
