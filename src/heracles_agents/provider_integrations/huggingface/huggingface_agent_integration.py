# ruff: noqa: F811

import copy

from collections.abc import Callable
from plum import dispatch

from heracles_agents.agent_functions import (
    build_custom_tool_prompt,
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import NormalizedMessage
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.huggingface.huggingface_client import (
    HuggingFaceClientConfig,
)
from heracles_agents.provider_integrations.huggingface.prompt_rendering import (
    render_huggingface_prompt,
)
import heracles_agents.provider_integrations.huggingface.token_counting  # noqa: F401
from heracles_agents.token_utils import count_text_tokens


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[HuggingFaceClientConfig]):
    p = copy.deepcopy(prompt)
    if agent.agent_info.tool_interface == "custom":
        p.tool_description = build_custom_tool_prompt(agent.agent_info.tools.values())
    return render_huggingface_prompt(p)


@dispatch
def iterate_messages(agent: LlmAgent[HuggingFaceClientConfig], response: list):
    yield from response


@dispatch
def is_function_call(agent: LlmAgent[HuggingFaceClientConfig], message):
    return False


@dispatch
def call_function(agent: LlmAgent[HuggingFaceClientConfig], message: dict):
    tool_string = extract_tag("tool", message["content"])
    return call_custom_tool_from_string(agent.agent_info.tools, tool_string)


@dispatch
def make_tool_response(agent: LlmAgent[HuggingFaceClientConfig], message: dict, result):
    return {"role": "user", "content": f"Output of tool call: {result}"}


@dispatch
def extract_answer(
    agent: LlmAgent[HuggingFaceClientConfig],
    extractor: Callable,
    message: dict,
):
    return extractor(message["content"])


@dispatch
def normalize_message(agent: LlmAgent[HuggingFaceClientConfig], message: dict):
    return NormalizedMessage(
        kind="assistant_text",
        text=message["content"],
        raw=message,
    )


@dispatch
def count_message_tokens(agent: LlmAgent[HuggingFaceClientConfig], message: dict):
    return count_text_tokens(agent, message["content"])
