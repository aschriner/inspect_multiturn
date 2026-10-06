"""Core protocol types shared by the orchestrator and user simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from inspect_ai.model import Content
from inspect_ai.solver import TaskState
from inspect_ai.util import StoreModel
from pydantic import Field


@dataclass
class UserMessage:
    """The simulated user sends a message to the target.

    Attributes:
        content: Message content.
        metadata: Simulator-private data for this turn (e.g. its reasoning).
            Recorded in `ConversationState.simulated`; never added to the
            transcript.
    """

    content: str | list[Content]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stop:
    """The simulated user ends the conversation.

    Attributes:
        reason: Short machine-readable reason (e.g. `"goal_met"`, `"gave_up"`,
            `"script_exhausted"`). Recorded as `ConversationState.stop_reason`.
        metadata: Additional detail, recorded as `ConversationState.stop_metadata`.
    """

    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


UserAction = UserMessage | Stop
"""An action returned by a `UserSimulator`."""


@dataclass(frozen=True)
class TurnInfo:
    """Where the conversation is when a simulator is asked for its next action.

    Attributes:
        turn: 0-based index of the turn being produced. When the conversation opens
            with the dataset input, that input is turn 0.
        user_turn: 0-based count of messages this simulator has produced so far,
            i.e. the index of the message being produced.
    """

    turn: int
    user_turn: int


class UserSimulator(Protocol):
    """Produces the simulated user's next action.

    Simulators may read anything on the `TaskState` (messages, metadata, sample
    id, epoch, store, ...) but must not modify the transcript: `converse()` is
    its only writer, and raises if a simulator changes it. What a simulator
    shows its own model is its decision (see `llm_user(visible_to_user=...)`).
    """

    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction:
        """Return the next user action.

        Args:
            state: The sample's current state. Treat as read-only.
            turn: Position in the conversation.
        """
        ...


class ConversationState(StoreModel):
    """Conversation outcome, stored per sample for use by scorers.

    Attributes:
        turns: Number of completed turns (user message plus target response).
        stop_reason: Why the conversation ended: a simulator `Stop.reason`,
            `"max_turns"`, `"target_condition"`, or `"limit"` if an Inspect
            sample limit ended it.
        stop_metadata: `Stop.metadata` from the simulator, if it stopped.
        simulated: Transcript messages written by the simulator, in order, as
            message id -> `UserMessage.metadata`. The transcript itself carries
            no marker, so test `message.id in simulated` to identify them.
    """

    turns: int = 0
    stop_reason: str | None = None
    stop_metadata: dict[str, Any] = Field(default_factory=dict)
    simulated: dict[str, dict[str, Any]] = Field(default_factory=dict)
