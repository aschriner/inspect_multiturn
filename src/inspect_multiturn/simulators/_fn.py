from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from inspect_ai.solver import TaskState

from .._types import TurnInfo, UserAction, UserMessage

UserFn = Callable[
    [TaskState, TurnInfo], "UserAction | str | Awaitable[UserAction | str]"
]


def fn_user(fn: UserFn) -> _FnUser:
    """A simulator backed by a plain function, for rule-based branching.

    Args:
        fn: Sync or async function taking the `TaskState` and a `TurnInfo` and
            returning a `UserAction`; a `str` is shorthand for `UserMessage(str)`.
            It must not modify the transcript.
    """
    return _FnUser(fn)


class _FnUser:
    def __init__(self, fn: UserFn) -> None:
        self._fn = fn

    async def __call__(self, state: TaskState, turn: TurnInfo) -> UserAction:
        result = self._fn(state, turn)
        if inspect.isawaitable(result):
            result = await result
        return UserMessage(result) if isinstance(result, str) else result
