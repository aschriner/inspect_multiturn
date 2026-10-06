import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ModelOutput,
)

from inspect_multiturn import (
    Stop,
    TurnInfo,
    UserMessage,
    fn_user,
    llm_user,
    scripted_user,
)
from inspect_multiturn.simulators._llm import flip_roles

from .conftest import make_state, mock_model, text_output


def turn(user_turn: int = 0) -> TurnInfo:
    return TurnInfo(turn=user_turn, user_turn=user_turn)


# scripted_user


async def test_scripted_user_replays_turns_then_stops():
    user = scripted_user(["one", "two"])
    state = make_state()
    assert await user(state, turn(0)) == UserMessage("one")
    assert await user(state, turn(1)) == UserMessage("two")
    assert await user(state, turn(2)) == Stop("script_exhausted")


async def test_scripted_user_reads_metadata_turns():
    user = scripted_user()
    state = make_state(metadata={"turns": ["from metadata"]})
    assert await user(state, turn()) == UserMessage("from metadata")


async def test_scripted_user_rejects_bad_metadata():
    with pytest.raises(ValueError, match='metadata\\["turns"\\]'):
        await scripted_user()(make_state(metadata={"turns": "not a list"}), turn())


# fn_user


async def test_fn_user_accepts_sync_str():
    user = fn_user(lambda state, t: f"turn {t.turn} of sample {state.sample_id}")
    assert await user(make_state(), turn(3)) == UserMessage("turn 3 of sample 1")


async def test_fn_user_accepts_async_action():
    async def decide(state, t):
        return Stop("done", {"why": "test"})

    assert await fn_user(decide)(make_state(), turn()) == Stop("done", {"why": "test"})


# flip_roles


def test_flip_roles_swaps_perspective():
    flipped = flip_roles(
        [
            ChatMessageUser(content="I'm the user"),
            ChatMessageAssistant(content="I'm the target"),
        ]
    )
    assert [(m.role, m.text) for m in flipped] == [
        ("user", "[The conversation begins.]"),
        ("assistant", "I'm the user"),
        ("user", "I'm the target"),
    ]


def test_flip_roles_empty_history_prompts_opening():
    [message] = flip_roles([])
    assert message.role == "user"
    assert "opening message" in message.text


def test_flip_roles_ends_with_user_message():
    flipped = flip_roles([ChatMessageAssistant(content="target spoke first")])
    assert flipped[-1].role == "user"
    assert [m.role for m in flip_roles([ChatMessageUser(content="x")])] == [
        "user",
        "assistant",
        "user",
    ]


# llm_user


async def test_llm_user_returns_message_and_uses_goal_and_persona():
    seen = {}

    def respond(input, tools, tool_choice, config):
        seen["input"] = input
        seen["tools"] = [t.name for t in tools]
        return text_output("  can u help me  ")

    user = llm_user(model=mock_model(respond), persona="A tired parent.")
    action = await user(
        make_state(
            [
                ChatMessageSystem(content="Target's secret system prompt."),
                ChatMessageAssistant(content="How can I help?"),
            ],
            metadata={"goal": "Get a refund."},
        ),
        turn(),
    )

    assert action == UserMessage("can u help me")
    system = seen["input"][0]
    assert isinstance(system, ChatMessageSystem)
    assert "Get a refund." in system.text
    assert "A tired parent." in system.text
    assert [m.role for m in seen["input"][1:]] == ["user"]
    assert "secret" not in str(seen["input"])
    assert seen["tools"] == ["end_conversation"]


async def test_llm_user_end_tool_goal_met():
    output = ModelOutput.for_tool_call(
        "mockllm",
        "end_conversation",
        {"goal_met": True, "explanation": "It agreed."},
    )
    user = llm_user(model=mock_model([output]), goal="g")
    action = await user(make_state(), turn())
    assert action == Stop("goal_met", {"explanation": "It agreed."})


async def test_llm_user_end_tool_gave_up():
    output = ModelOutput.for_tool_call(
        "mockllm",
        "end_conversation",
        {"goal_met": False, "explanation": "No luck."},
    )
    user = llm_user(model=mock_model([output]), goal="g")
    action = await user(make_state(), turn())
    assert action == Stop("gave_up", {"explanation": "No luck."})


async def test_llm_user_records_reasoning():
    output = text_output("hey")
    output.choices[0].message.content = [
        ContentReasoning(reasoning="try flattery"),
        ContentText(text="hey"),
    ]
    user = llm_user(model=mock_model([output]), goal="g")
    action = await user(make_state(), turn())
    assert action == UserMessage("hey", {"reasoning": "try flattery"})


async def test_llm_user_empty_response_stops():
    user = llm_user(model=mock_model([text_output("   ")]), goal="g")
    action = await user(make_state(), turn())
    assert action == Stop("empty_response")


async def test_llm_user_requires_goal():
    user = llm_user(model=mock_model())
    with pytest.raises(ValueError, match="requires a goal"):
        await user(make_state(), turn())


async def test_llm_user_custom_visibility():
    seen = {}

    def respond(input, tools, tool_choice, config):
        seen["input"] = input
        return text_output("ok")

    user = llm_user(
        model=mock_model(respond), goal="g", visible_to_user=lambda messages: []
    )
    await user(make_state([ChatMessageAssistant(content="hidden")]), turn())
    assert "hidden" not in str(seen["input"])
