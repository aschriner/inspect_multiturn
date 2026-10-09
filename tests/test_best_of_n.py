import itertools

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ModelOutput
from inspect_ai.solver import TaskState

from inspect_multiturn import (
    ConversationState,
    LLMSelectorMetadata,
    Selection,
    Stop,
    TurnInfo,
    UserAction,
    UserMessage,
    best_of_n_user,
    converse,
    fn_user,
    llm_selector,
    scripted_user,
)

from .conftest import make_state, mock_model, text_output


def turn(user_turn: int = 0) -> TurnInfo:
    return TurnInfo(turn=user_turn, user_turn=user_turn)


def drafts(*actions: UserAction):
    """A generator that returns `actions` in order, one per call."""
    remaining = iter(actions)
    return fn_user(lambda state, turn: next(remaining))


def choose(index: int, seen: list | None = None, **metadata):
    async def selector(state, turn, candidates):
        if seen is not None:
            seen.append(candidates)
        return Selection(index, metadata)

    return selector


def select_call(candidate, rationale: str = "most natural") -> ModelOutput:
    return ModelOutput.for_tool_call(
        "mockllm",
        "select_candidate",
        {"candidate": candidate, "rationale": rationale},
    )


# best_of_n_user


async def test_best_of_n_sends_selected_candidate_and_records_all():
    seen: list = []
    user = best_of_n_user(
        drafts(UserMessage("a", {"x": 1}), UserMessage("b", {"y": 2}), Stop("done")),
        n=3,
        selector=choose(1, seen, rationale="r"),
    )
    action = await user(make_state(), turn())

    assert seen == [
        [UserMessage("a", {"x": 1}), UserMessage("b", {"y": 2}), Stop("done")]
    ]
    assert action == UserMessage(
        "b",
        {
            "y": 2,
            "candidates": [
                {"content": "a", "metadata": {"x": 1}},
                {"content": "b", "metadata": {"y": 2}},
                {"stop": "done", "metadata": {}},
            ],
            "selected": 1,
            "selection": {"rationale": "r"},
        },
    )


async def test_best_of_n_can_select_a_stop():
    user = best_of_n_user(
        drafts(UserMessage("a"), Stop("end_conversation", {"text": "bye"})),
        n=2,
        selector=choose(1),
    )
    action = await user(make_state(), turn())
    assert isinstance(action, Stop)
    assert action.reason == "end_conversation"
    assert action.metadata["text"] == "bye"
    assert action.metadata["selected"] == 1
    assert "selection" not in action.metadata


async def test_best_of_n_passes_state_and_turn_to_generator_and_selector():
    calls: list = []

    def generate(state, t):
        calls.append(("generator", state, t))
        return "hi"

    async def selector(state, t, candidates):
        calls.append(("selector", state, t))
        return Selection(0)

    state = make_state()
    await best_of_n_user(fn_user(generate), n=2, selector=selector)(state, turn(3))
    assert calls == [
        ("generator", state, turn(3)),
        ("generator", state, turn(3)),
        ("selector", state, turn(3)),
    ]


async def test_best_of_n_with_one_candidate_skips_selector():
    async def selector(state, turn, candidates):
        raise AssertionError("selector called")

    user = best_of_n_user(drafts(UserMessage("only")), n=1, selector=selector)
    action = await user(make_state(), turn())
    assert action == UserMessage(
        "only",
        {"candidates": [{"content": "only", "metadata": {}}], "selected": 0},
    )


async def test_best_of_n_rejects_out_of_range_selection():
    user = best_of_n_user(
        drafts(UserMessage("a"), UserMessage("b")), n=2, selector=choose(2)
    )
    with pytest.raises(ValueError, match="index 2"):
        await user(make_state(), turn())


def test_best_of_n_requires_positive_n():
    with pytest.raises(ValueError, match="at least 1"):
        best_of_n_user(fn_user(lambda s, t: "x"), n=0)


def test_best_of_n_with_llm_selector_runs_in_converse(run_solver):
    counter = itertools.count()

    def generate(state: TaskState, t: TurnInfo) -> str:
        return f"turn {t.user_turn} draft {next(counter) % 2}"

    user = best_of_n_user(
        fn_user(generate),
        n=2,
        selector=llm_selector(model=mock_model([select_call(2)] * 2)),
    )
    sample = run_solver(
        converse(user, max_turns=2, first_turn="simulator"),
        Sample(input=[], metadata={"goal": "g"}),
    )

    users = [m.text for m in sample.messages if m.role == "user"]
    assert users == ["turn 0 draft 1", "turn 1 draft 1"]
    conv = sample.store_as(ConversationState)
    assert [m["selected"] for m in conv.simulated.values()] == [1, 1]


# llm_selector


async def test_llm_selector_prompts_with_goal_criteria_conversation_and_candidates():
    seen = {}

    def respond(input, tools, tool_choice, config):
        seen.update(input=input, tools=tools, tool_choice=tool_choice)
        return select_call(2, "sounds human")

    selector = llm_selector(model=mock_model(respond), criteria="Be terse.")
    state = make_state(
        [
            ChatMessageSystem(content="Target's secret system prompt."),
            ChatMessageAssistant(content="How can I help?"),
        ],
        metadata={"goal": "Get a refund."},
    )
    selection = await selector(
        state,
        turn(),
        [UserMessage("Dear assistant,"), UserMessage("refund pls"), Stop("done")],
    )

    assert selection == Selection(1, {"rationale": "sounds human"})
    [message] = seen["input"]
    prompt = message.text
    assert "Get a refund." in prompt
    assert "Be terse." in prompt
    assert "ASSISTANT: How can I help?" in prompt
    assert '<candidate number="2">refund pls</candidate>' in prompt
    assert "Ends the conversation" in prompt and "done" in prompt
    assert "secret" not in prompt
    assert [t.name for t in seen["tools"]] == ["select_candidate"]
    assert seen["tool_choice"].name == "select_candidate"


async def test_llm_selector_reads_criteria_from_metadata():
    seen = {}

    def respond(input, tools, tool_choice, config):
        seen["prompt"] = input[0].text
        return select_call("1")

    selector = llm_selector(model=mock_model(respond))
    state = make_state(metadata={"selection_criteria": "Pick the angriest."})
    selection = await selector(state, turn(), [UserMessage("a"), UserMessage("b")])
    assert selection.index == 0
    assert "Pick the angriest." in seen["prompt"]
    assert "(not specified)" in seen["prompt"]


@pytest.mark.parametrize(
    ("output", "error"),
    [
        (select_call(3), "invalid candidate 3"),
        (select_call(0), "invalid candidate 0"),
        (select_call("two"), "invalid candidate 'two'"),
        (text_output("I pick 2"), "no select_candidate call"),
    ],
)
async def test_llm_selector_falls_back_to_first_candidate(output, error):
    selector = llm_selector(model=mock_model([output]))
    selection = await selector(
        make_state(), turn(), [UserMessage("a"), UserMessage("b")]
    )
    assert selection == Selection(0, {"error": error})


async def test_llm_selector_requires_model():
    with pytest.raises(ValueError, match="user_selector"):
        await llm_selector()(make_state(), turn(), [UserMessage("a"), UserMessage("b")])


def test_llm_selector_metadata_model():
    assert LLMSelectorMetadata.model_validate({"goal": "g", "other": 1}) == (
        LLMSelectorMetadata(goal="g")
    )


async def test_best_of_n_works_with_scripted_generator_and_llm_selector():
    user = best_of_n_user(
        scripted_user(["same"]),
        n=2,
        selector=llm_selector(model=mock_model([select_call(1)])),
    )
    action = await user(make_state(), turn())
    assert isinstance(action, UserMessage)
    assert action.content == "same"
    assert action.metadata["selection"] == {"rationale": "most natural"}
