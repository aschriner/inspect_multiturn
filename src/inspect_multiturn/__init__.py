"""An Inspect extension for multi-turn evaluations."""

from importlib.metadata import PackageNotFoundError, version

from ._conversation import converse
from ._stop import StopCondition, tool_called
from ._types import (
    ConversationState,
    Stop,
    TurnInfo,
    UserAction,
    UserMessage,
    UserSimulator,
)
from .simulators import (
    LLMUserMetadata,
    ScriptedUserMetadata,
    UserLMMetadata,
    ViewFn,
    default_user_view,
    fn_user,
    llm_user,
    scripted_user,
    userlm_user,
)

try:
    __version__ = version("inspect-multiturn")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "ConversationState",
    "LLMUserMetadata",
    "ScriptedUserMetadata",
    "Stop",
    "StopCondition",
    "TurnInfo",
    "UserAction",
    "UserLMMetadata",
    "UserMessage",
    "UserSimulator",
    "ViewFn",
    "__version__",
    "converse",
    "default_user_view",
    "fn_user",
    "llm_user",
    "scripted_user",
    "tool_called",
    "userlm_user",
]
