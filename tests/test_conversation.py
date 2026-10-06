import re

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import AgentState, agent
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.solver import TaskState, use_tools
from inspect_ai.tool import tool

from inspect_multiturn import (
    ConversationState,
    Stop,
    TurnInfo,
    UserMessage,
    converse,
    fn_user,
    llm_user,
    scripted_user,
    tool_called,
)

from .conftest import mock_model, text_output

REPLY = "Default output from mockllm/model"


def roles_and_text(sample):
    return [(m.role, m.text) for m in sample.messages]


def test_dataset_first_turn_then_script(run_solver):
    solver = converse(user=scripted_user(["second", "third"]))
    sample = run_solver(solver, Sample(input="first"))

    assert roles_and_text(sample) == [
        ("user", "first"),
        ("assistant", REPLY),
        ("user", "second"),
        ("assistant", REPLY),
        ("user", "third"),
        ("assistant", REPLY),
    ]
    assert sample.output.completion == REPLY
    conv = sample.store_as(ConversationState)
    assert conv.turns == 3
    assert conv.stop_reason == "script_exhausted"
    assert conv.simulated == {sample.messages[2].id: {}, sample.messages[4].id: {}}
    assert all(m.metadata is None for m in sample.messages)


def test_max_turns_counts_dataset_turn(run_solver):
    solver = converse(user=scripted_user(["a", "b", "c", "d"]), max_turns=2)
    sample = run_solver(solver, Sample(input="first"))

    conv = sample.store_as(ConversationState)
    assert conv.turns == 2
    assert conv.stop_reason == "max_turns"
    assert len(sample.messages) == 4


def test_simulator_opens_conversation(run_solver):
    turns: list[TurnInfo] = []

    def user(state: TaskState, turn: TurnInfo):
        turns.append(turn)
        return "hello" if turn.user_turn == 0 else Stop("done", {"k": "v"})

    sample = run_solver(
        converse(user=fn_user(user), first_turn="simulator"),
        Sample(input=[]),
    )

    assert turns == [TurnInfo(0, 0), TurnInfo(1, 1)]
    conv = sample.store_as(ConversationState)
    assert conv.turns == 1
    assert conv.stop_reason == "done"
    assert conv.stop_metadata == {"k": "v"}


def test_simulator_reads_full_task_state(run_solver):
    seen = {}

    def user(state: TaskState, turn: TurnInfo):
        seen["roles"] = [m.role for m in state.messages]
        seen["metadata"] = state.metadata
        seen["sample_id"] = state.sample_id
        seen["epoch"] = state.epoch
        return Stop("done")

    run_solver(
        converse(user=fn_user(user)),
        Sample(
            id="s1",
            input=[
                ChatMessageSystem(content="system prompt"),
                ChatMessageUser(content="first"),
            ],
            metadata={"goal": "g"},
        ),
    )

    assert seen == {
        "roles": ["system", "user", "assistant"],
        "metadata": {"goal": "g"},
        "sample_id": "s1",
        "epoch": 1,
    }


def test_simulator_cannot_modify_transcript(run_solver):
    def user(state: TaskState, turn: TurnInfo):
        state.messages.append(ChatMessageUser(content="sneaky"))
        return "hi"

    sample = run_solver(converse(user=fn_user(user)), Sample(input="x"))
    assert sample.error is not None
    assert "modified the transcript" in sample.error.message


def test_user_message_metadata_recorded(run_solver):
    def user(state: TaskState, turn: TurnInfo):
        if turn.user_turn == 0:
            return UserMessage("next", {"reasoning": "push harder"})
        return Stop("done")

    sample = run_solver(converse(user=fn_user(user)), Sample(input="x"))

    assert sample.messages[2].metadata is None
    conv = sample.store_as(ConversationState)
    assert conv.simulated == {sample.messages[2].id: {"reasoning": "push harder"}}


def test_stop_when_target_calls_tool(run_solver):
    @tool
    def transfer_to_human():
        async def execute() -> str:
            """Transfer the user to a human agent."""
            return "Transferred."

        return execute

    target_model = mock_model(
        [
            ModelOutput.for_tool_call("mockllm", "transfer_to_human", {}),
            text_output("A human will be with you shortly."),
        ]
    )
    solver = [
        use_tools(transfer_to_human()),
        converse(
            user=scripted_user(["more"]),
            stop_when=tool_called("transfer_to_human"),
        ),
    ]
    sample = run_solver(solver, Sample(input="get me a human"), model=target_model)

    conv = sample.store_as(ConversationState)
    assert conv.stop_reason == "target_condition"
    assert conv.turns == 1
    assert [m.role for m in sample.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_custom_target_agent(run_solver):
    seen: list[list[str]] = []

    @agent
    def echo_target():
        async def execute(state: AgentState) -> AgentState:
            seen.append([m.role for m in state.messages])
            reply = ChatMessageAssistant(content=f"echo: {state.messages[-1].text}")
            state.messages.append(reply)
            return state

        return execute

    sample = run_solver(
        converse(scripted_user(["second"]), target=echo_target()),
        Sample(input="first"),
    )

    assert roles_and_text(sample) == [
        ("user", "first"),
        ("assistant", "echo: first"),
        ("user", "second"),
        ("assistant", "echo: second"),
    ]
    assert seen == [["user"], ["user", "assistant", "user"]]
    assert sample.output.completion == "echo: second"
    assert sample.store_as(ConversationState).turns == 2


def test_defaults_to_llm_user_via_model_role(tmp_path):
    user_model = mock_model(
        [
            text_output("pls give me a refund"),
            ModelOutput.for_tool_call(
                "mockllm",
                "end_conversation",
                {"goal_met": False, "explanation": "It refused."},
            ),
        ]
    )
    task = Task(
        dataset=[Sample(input="hi", metadata={"goal": "Get a refund."})],
        solver=converse(),
    )
    [log] = eval(
        task,
        model="mockllm/model",
        model_roles={"user": user_model},
        display="none",
        log_dir=str(tmp_path),
    )
    sample = log.samples[0]

    assert sample.messages[2].text == "pls give me a refund"
    conv = sample.store_as(ConversationState)
    assert conv.turns == 2
    assert conv.stop_reason == "gave_up"
    assert conv.stop_metadata == {"explanation": "It refused."}


def test_llm_user_requires_explicit_model(run_solver):
    sample = run_solver(converse(user=llm_user(goal="g")), Sample(input="hi"))
    assert sample.error is not None
    assert "needs an explicit model" in sample.error.message


def test_limit_records_stop_reason(run_solver):
    solver = converse(user=scripted_user(["a", "b", "c"]))
    sample = run_solver(solver, Sample(input="first"), message_limit=3)

    assert sample.limit is not None
    assert sample.store_as(ConversationState).stop_reason == "limit"


@pytest.mark.parametrize(
    ("first_turn", "sample_input", "message"),
    [
        ("dataset", [], "to end with a user message"),
        ("simulator", "hi", "not to end with a user message"),
    ],
)
def test_first_turn_validation(run_solver, first_turn, sample_input, message):
    sample = run_solver(
        converse(user=scripted_user([]), first_turn=first_turn),
        Sample(input=sample_input),
    )
    assert sample.error is not None
    assert re.search(message, sample.error.message)


@pytest.mark.parametrize(
    ("sample_input", "opener"),
    [
        ("from dataset", "from dataset"),
        ([], "from simulator"),
        ([ChatMessageSystem(content="sys")], "from simulator"),
    ],
)
def test_auto_first_turn(run_solver, sample_input, opener):
    sample = run_solver(
        converse(scripted_user(["from simulator"]), max_turns=1),
        Sample(input=sample_input),
    )
    assert sample.error is None
    [first_user] = [m for m in sample.messages if m.role == "user"]
    assert first_user.text == opener


def test_first_turn_must_be_known():
    with pytest.raises(ValueError, match="first_turn"):
        converse(scripted_user([]), first_turn="user")  # type: ignore[arg-type]


def test_target_is_keyword_only():
    with pytest.raises(TypeError):
        converse(scripted_user([]), None)  # type: ignore[misc]


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError, match="max_turns"):
        converse(user=scripted_user([]), max_turns=0)
