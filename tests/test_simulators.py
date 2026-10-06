import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
)

from inspect_multiturn import (
    LLMUserMetadata,
    ScriptedUserMetadata,
    Stop,
    TurnInfo,
    UserLMMetadata,
    UserMessage,
    fn_user,
    llm_user,
    scripted_user,
    userlm_user,
)
from inspect_multiturn.simulators import _userlm
from inspect_multiturn.simulators._llm import flip_roles
from inspect_multiturn.simulators._userlm import render_userlm_prompt

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


async def test_scripted_user_accepts_json_turns_from_csv():
    user = scripted_user()
    state = make_state(metadata={"turns": '["a", "b"]'})
    assert await user(state, turn(1)) == UserMessage("b")


@pytest.mark.parametrize(
    ("turns", "key"),
    [
        (None, 'metadata\\["turns"\\]'),
        (["ok", 3], 'metadata\\["turns"\\]\\[1\\]'),
        ('{"a": 1}', 'metadata\\["turns"\\]'),
    ],
)
async def test_scripted_user_names_invalid_key(turns, key):
    metadata = {} if turns is None else {"turns": turns}
    with pytest.raises(ValueError, match=key):
        await scripted_user()(make_state(metadata=metadata), turn())


def test_scripted_user_metadata_model():
    assert ScriptedUserMetadata.model_validate({"turns": '["x"]', "other": 1}) == (
        ScriptedUserMetadata(turns=["x"])
    )


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


async def test_llm_user_rejects_empty_metadata_goal():
    user = llm_user(model=mock_model())
    with pytest.raises(ValueError, match='metadata\\["goal"\\]'):
        await user(make_state(metadata={"goal": ""}), turn())


async def test_llm_user_arguments_override_metadata():
    seen = {}

    def respond(input, tools, tool_choice, config):
        seen["system"] = input[0].text
        return text_output("ok")

    user = llm_user(model=mock_model(respond), persona="Arg persona.")
    state = make_state(metadata={"goal": "Meta goal.", "persona": "Meta persona."})
    await user(state, turn())
    assert "Arg persona." in seen["system"]
    assert "Meta goal." in seen["system"]
    assert "Meta persona." not in seen["system"]


def test_llm_user_metadata_model():
    metadata = LLMUserMetadata.model_validate({"goal": "g", "behavior": "b"})
    assert metadata == LLMUserMetadata(goal="g", persona=None)


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


# userlm_user


def recording_model(*replies: str, seen: list | None = None):
    outputs = iter(replies)

    def respond(input, tools, tool_choice, config):
        if seen is not None:
            seen.append({"input": input, "tools": tools, "config": config})
        return text_output(next(outputs))

    return mock_model(respond)


async def test_userlm_user_sends_intent_and_unflipped_history():
    seen: list = []
    user = userlm_user(model=recording_model("  so can i get it back  ", seen=seen))
    state = make_state(
        [
            ChatMessageSystem(content="Target's secret system prompt."),
            ChatMessageUser(content="i need a refund"),
            ChatMessageAssistant(content="What's the order number?"),
        ],
        metadata={"goal": "You are a user who wants a refund."},
    )

    assert await user(state, turn(1)) == UserMessage("so can i get it back")
    [call] = seen
    assert [(m.role, m.text) for m in call["input"]] == [
        ("system", "You are a user who wants a refund."),
        ("user", "i need a refund"),
        ("assistant", "What's the order number?"),
    ]
    assert call["tools"] == []
    assert call["config"].temperature == 1.0
    assert call["config"].extra_body == {"add_generation_prompt": True}


@pytest.mark.parametrize("reply", ["<|endconversation|>", "   "])
async def test_userlm_user_end_token_stops(reply):
    user = userlm_user(model=recording_model(reply), goal="g")
    assert await user(make_state(), turn()) == Stop("end_conversation")


async def test_userlm_user_regenerates_rejected_drafts():
    user = userlm_user(
        model=recording_model("ok", "i need a refund", "fine, what are my options"),
        goal="g",
    )
    state = make_state([ChatMessageUser(content="I need a  refund")])
    action = await user(state, turn(1))
    assert action == UserMessage(
        "fine, what are my options",
        {
            "rejected": [
                {"text": "ok", "reason": "too_short"},
                {"text": "i need a refund", "reason": "repeated"},
            ]
        },
    )


async def test_userlm_user_stops_when_retries_exhausted():
    user = userlm_user(
        model=recording_model("one two three four", "five six seven eight"),
        goal="g",
        max_words=3,
        max_retries=1,
    )
    action = await user(make_state(), turn())
    assert isinstance(action, Stop)
    assert action.reason == "guardrails_exhausted"
    assert [r["reason"] for r in action.metadata["rejected"]] == ["too_long"] * 2


async def test_userlm_user_config_overrides_defaults():
    seen: list = []
    user = userlm_user(
        model=recording_model("hello there friend", seen=seen),
        goal="g",
        config=GenerateConfig(temperature=0.2),
    )
    await user(make_state(), turn())
    assert seen[0]["config"].temperature == 0.2
    assert seen[0]["config"].top_p == 0.8


# Verbatim from https://huggingface.co/microsoft/UserLM-8b/blob/main/chat_template.jinja
USERLM_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>' }}\n{{ message['content'] }}<|eot_id|>{% endfor %}"
    "{{ '<|start_header_id|>user<|end_header_id|>' }}"
)


def test_render_userlm_prompt_matches_model_chat_template():
    jinja2 = pytest.importorskip("jinja2")
    messages = [
        ChatMessageSystem(content="You are a user who wants a refund."),
        ChatMessageUser(content="i need a refund"),
        ChatMessageAssistant(content="What's the order number?"),
    ]
    expected = jinja2.Template(USERLM_CHAT_TEMPLATE).render(
        messages=[{"role": m.role, "content": m.text} for m in messages]
    )
    assert render_userlm_prompt(messages) == expected
    assert expected.endswith("<|start_header_id|>user<|end_header_id|>")


async def test_userlm_user_sends_raw_prompt_to_completions_providers(monkeypatch):
    monkeypatch.setattr(_userlm, "_uses_raw_prompt", lambda model: True)
    seen: list = []
    user = userlm_user(model=recording_model("what about refunds", seen=seen))
    state = make_state(
        [ChatMessageAssistant(content="How can I help?")],
        metadata={"goal": "You want a refund."},
    )

    assert await user(state, turn()) == UserMessage("what about refunds")
    [call] = seen
    [message] = call["input"]
    assert message.role == "user"
    assert message.text == render_userlm_prompt(
        [
            ChatMessageSystem(content="You want a refund."),
            ChatMessageAssistant(content="How can I help?"),
        ]
    )
    assert call["config"].max_tokens == 1024
    assert call["config"].extra_body is None


async def test_userlm_user_requires_goal():
    with pytest.raises(ValueError, match="requires a goal"):
        await userlm_user(model=mock_model())(make_state(), turn())


def test_userlm_user_metadata_model():
    metadata = UserLMMetadata.model_validate({"goal": "g", "persona": "p"})
    assert metadata == UserLMMetadata(goal="g")
