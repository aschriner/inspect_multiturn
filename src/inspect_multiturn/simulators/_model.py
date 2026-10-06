from __future__ import annotations

from inspect_ai.model import Model, get_model, model_roles


def resolve_user_model(
    model: str | Model | None, simulator: str, example: str, hint: str = ""
) -> Model:
    """Return the simulator's model: `model` if given, else the `user` role.

    Never falls back to the model being evaluated.
    """
    if model is not None:
        return get_model(model)
    if "user" not in model_roles():
        raise ValueError(
            f"{simulator}() needs an explicit model: pass {simulator}(model=...) or "
            f"set the user model role (e.g. --model-role user={example}).{hint}"
        )
    return get_model(role="user")
