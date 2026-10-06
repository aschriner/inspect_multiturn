"""Inspect extension registry.

Inspect imports this module via the ``inspect_ai`` entry point declared in
``pyproject.toml``. Import every decorated object (``@task``, ``@solver``,
``@scorer``, ``@modelapi``, ``@sandboxenv``, ...) here so that it is registered
and can be referenced by name, e.g. ``inspect_multiturn/my_solver``.
"""

from ._conversation import converse
