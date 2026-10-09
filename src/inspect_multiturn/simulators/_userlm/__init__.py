"""`userlm_user()`: a simulator backed by a user language model such as UserLM-8b."""

from ._simulator import UserLMMetadata, userlm_user
from ._template import render_userlm_prompt

__all__ = [
    "UserLMMetadata",
    "render_userlm_prompt",
    "userlm_user",
]
