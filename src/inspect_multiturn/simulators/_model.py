from __future__ import annotations

from inspect_ai.model import Model, get_model, model_roles


def resolve_user_model(
    model: str | Model | None,
    simulator: str,
    example: str,
    hint: str = "",
    role: str = "user",
) -> Model:
    """Return the simulator's model: `model` if given, else the `role` model role.

    Never falls back to the model being evaluated.
    """
    if model is not None:
        return get_model(model)
    if role not in model_roles():
        raise ValueError(
            f"{simulator}() needs an explicit model: pass {simulator}(model=...) or "
            f"set the {role} model role (e.g. --model-role {role}={example}).{hint}"
        )
    return get_model(role=role)
