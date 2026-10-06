from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalSample
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    Model,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Solver, TaskState
from inspect_ai.tool import ToolChoice, ToolInfo

MockFn = Callable[
    [list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig], ModelOutput
]


def mock_model(outputs: Sequence[ModelOutput] | MockFn | None = None) -> Model:
    """A fresh mockllm model, optionally with scripted outputs."""
    if outputs is None:
        return get_model("mockllm/model", memoize=False)
    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


def text_output(content: str) -> ModelOutput:
    return ModelOutput.from_content(model="mockllm", content=content)


def make_state(
    messages: Sequence[ChatMessage] = (), metadata: dict[str, Any] | None = None
) -> TaskState:
    """A TaskState for calling simulators directly, outside an eval."""
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input=list(messages),
        messages=list(messages),
        metadata=metadata or {},
    )


@pytest.fixture
def run_solver(tmp_path: Path) -> Callable[..., EvalSample]:
    """Run a solver on a single sample and return that sample."""

    def run(
        solver: Solver | list[Solver],
        sample: Sample,
        model: str | Model = "mockllm/model",
        **task_args: Any,
    ) -> EvalSample:
        task = Task(dataset=[sample], solver=solver, **task_args)
        [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
        assert log.samples is not None
        return log.samples[0]

    return run
