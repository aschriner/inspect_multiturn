from __future__ import annotations

import logging

from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelName,
)

logger = logging.getLogger(__name__)

# Inspect's completions providers default to max_tokens=1 (for perplexity evals).
_COMPLETIONS_MAX_TOKENS = 512

# Chat providers that load UserLM's own chat template from the model repo.
_LOCAL_TEMPLATE_APIS = {"vllm", "sglang", "hf"}

# Of those, the ones that continue a final assistant message instead of opening a
# new turn, unless the request sets add_generation_prompt.
_CONTINUES_FINAL_MESSAGE_APIS = {"vllm", "sglang"}

_warned_chat_models: set[str] = set()


def render_userlm_prompt(messages: list[ChatMessage]) -> str:
    """Render messages with UserLM-8b's chat template, ending on an open user turn.

    Mirrors the template in the model repo (`chat_template.jinja`), for
    endpoints that take a raw prompt.

    Args:
        messages: The intent as a system message, then the conversation with
            the simulated user's turns as `user` and the target's as `assistant`.
    """
    turns = "".join(
        f"<|start_header_id|>{m.role}<|end_header_id|>\n{m.text}<|eot_id|>"
        for m in messages
    )
    return f"{turns}<|start_header_id|>user<|end_header_id|>"


def request(
    model: Model, messages: list[ChatMessage], config: GenerateConfig
) -> tuple[list[ChatMessage], GenerateConfig]:
    """The input and config to send `model` for the next user turn."""
    api = _api(model)

    # Completions providers send the prompt as-is, with no chat template.
    if api.endswith("-completions"):
        prompt = ChatMessageUser(content=render_userlm_prompt(messages))
        if config.max_tokens is None:
            config = config.merge(GenerateConfig(max_tokens=_COMPLETIONS_MAX_TOKENS))
        return [prompt], config

    if api in _CONTINUES_FINAL_MESSAGE_APIS:
        # The prompt ends with the target's (assistant) reply, which these
        # providers would continue rather than start a new turn. UserLM's template
        # always opens a user turn once add_generation_prompt is set.
        extra_body = dict(config.extra_body or {})
        if not {"add_generation_prompt", "continue_final_message"} & extra_body.keys():
            extra_body["add_generation_prompt"] = True
            config = config.merge(GenerateConfig(extra_body=extra_body))
    elif api not in _LOCAL_TEMPLATE_APIS and str(model) not in _warned_chat_models:
        _warned_chat_models.add(str(model))
        logger.warning(
            f"userlm_user() is using the chat API of '{model}', which may not apply "
            "UserLM's chat template; if so, the model writes the assistant's side "
            "of the conversation. Prefer a completions provider, e.g. "
            "'openai-api-completions/<service>/<model>'."
        )
    return messages, config


def _api(model: Model) -> str:
    return ModelName(model).api
