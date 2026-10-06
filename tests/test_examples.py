import itertools
from pathlib import Path

from inspect_ai import eval

from inspect_multiturn import ConversationState

from .conftest import mock_model, text_output

ROOT = Path(__file__).parent.parent


def test_refund_policy_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # Inspect resolves task files relative to the cwd
    [log] = eval(
        "examples/refund_policy.py",
        model="mockllm/model",
        model_roles={"user": "mockllm/model"},
        task_args={"max_turns": 3},
        display="none",
        log_dir=str(tmp_path),
    )
    assert log.status == "success", log.error
    assert log.samples is not None
    for sample in log.samples:
        assert sample.messages[0].role == "system"
        conv = sample.store_as(ConversationState)
        assert sample.messages[1].id in conv.simulated
        assert sample.scores["behavior_elicited"].metadata == {
            "turns": 3,
            "stop_reason": "max_turns",
        }


def test_socratic_tutor_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    user_turns = itertools.count()

    def student(input, tools, tool_choice, config):
        if "Maya" in input[0].text:  # the simulator's system prompt holds the goal
            return text_output(f"is it 360 or what ({next(user_turns)})")
        return text_output(f"can u just tell me ({next(user_turns)})")

    target = mock_model([text_output("Hmm, would it be 288 or 360?")] * 9)
    [log] = eval(
        "examples/socratic_tutor.py",
        model=target,
        model_roles={"user": mock_model(student)},
        task_args={"max_turns": 3},
        display="none",
        log_dir=str(tmp_path),
    )
    assert log.status == "success", log.error
    assert log.samples is not None
    scores = {s.id: s.scores["answer_withheld"] for s in log.samples}
    assert {id: score.value for id, score in scores.items()} == {
        "bakery": "I",
        "reading": "C",
        "water-tank": "C",
    }
    assert scores["reading"].metadata["student_said_answer"] is True
    assert all(score.metadata["turns"] == 3 for score in scores.values())
