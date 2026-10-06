"""`llm_user()`: a model playing a goal-driven user."""

from ._history import flip_roles
from ._prompt import DEFAULT_USER_PROMPT
from ._simulator import LLMUserMetadata, llm_user
from ._view import ViewFn, default_user_view

__all__ = [
    "DEFAULT_USER_PROMPT",
    "LLMUserMetadata",
    "ViewFn",
    "default_user_view",
    "flip_roles",
    "llm_user",
]
