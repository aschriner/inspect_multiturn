from pathlib import Path

from inspect_ai import eval

from inspect_multiturn import ConversationState

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
