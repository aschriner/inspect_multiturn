from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall

from inspect_multiturn import default_user_view


def test_default_view_keeps_only_user_visible_content():
    user = ChatMessageUser(content="hi")
    messages = [
        ChatMessageSystem(content="secret system prompt"),
        user,
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="1", function="lookup", arguments={})],
        ),
        ChatMessageTool(content="tool result", tool_call_id="1", function="lookup"),
        ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning="private thoughts"),
                ContentText(text="Hello!"),
            ],
            tool_calls=[ToolCall(id="2", function="log", arguments={})],
        ),
    ]

    view = default_user_view(messages)

    assert [(m.role, m.text) for m in view] == [
        ("user", "hi"),
        ("assistant", "Hello!"),
    ]
    assert view[1].tool_calls is None
    assert view[0] is not user


def test_default_view_does_not_modify_input():
    message = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="r"), ContentText(text="t")]
    )
    default_user_view([message])
    assert len(message.content) == 2
