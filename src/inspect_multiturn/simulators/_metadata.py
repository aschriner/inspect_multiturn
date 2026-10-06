from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


def read_metadata(model: type[M], metadata: dict[str, Any], hint: str) -> M:
    """Validate sample metadata against a simulator's metadata model.

    Raises a `ValueError` naming each invalid key as `metadata["key"]`, followed
    by `hint` on how to fix it.
    """
    try:
        return model.model_validate(metadata)
    except ValidationError as ex:
        problems = "; ".join(
            f"{_key(error['loc'])}: {error['msg']}" for error in ex.errors()
        )
        raise ValueError(f"Invalid sample metadata ({problems}). {hint}") from None


def _key(loc: tuple[int | str, ...]) -> str:
    head, *rest = loc
    return f'metadata["{head}"]' + "".join(f"[{part!r}]" for part in rest)
