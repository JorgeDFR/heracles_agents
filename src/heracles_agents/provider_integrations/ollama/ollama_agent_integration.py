# ruff: noqa: F811

import copy
import logging

from collections.abc import Callable
from plum import dispatch

from ollama import ChatResponse, Message

from heracles_agents.agent_functions import (
    build_custom_tool_prompt,
    call_custom_tool_from_string,
    extract_tag,
    get_tool_function,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import NormalizedMessage
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.ollama.ollama_client import (
    OllamaClientConfig,
)
from heracles_agents.provider_integrations.ollama.prompt_rendering import (
    render_ollama_prompt,
)
import heracles_agents.provider_integrations.ollama.tool_rendering  # noqa: F401
import heracles_agents.provider_integrations.ollama.token_counting  # noqa: F401
from heracles_agents.token_utils import count_text_tokens

logger = logging.getLogger(__name__)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OllamaClientConfig]):
    p = copy.deepcopy(prompt)
    if agent.agent_info.tool_interface == "custom":
        p.tool_description = build_custom_tool_prompt(agent.agent_info.tools.values())
    return render_ollama_prompt(p)


@dispatch
def iterate_messages(agent: LlmAgent[OllamaClientConfig], response: ChatResponse):
    for m in response.message.tool_calls or []:
        yield m
    yield response.message


@dispatch
def is_function_call(agent: LlmAgent[OllamaClientConfig], tool_call: Message.ToolCall):
    return True


@dispatch
def is_function_call(agent: LlmAgent[OllamaClientConfig], tool_call):
    return False


@dispatch
def call_function(agent: LlmAgent[OllamaClientConfig], tool_call: Message.ToolCall):
    available_tools = agent.agent_info.tools
    name = tool_call.function.name
    return get_tool_function(available_tools, name)(**tool_call.function.arguments)


@dispatch
def call_function(agent: LlmAgent[OllamaClientConfig], tool_call: Message):
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_call.content)
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[OllamaClientConfig],
    tool_call_message: Message.ToolCall,
    result,
):
    m = {
        "role": "tool",
        "tool_name": tool_call_message.function.name,
        "content": str(result),
    }
    return m


@dispatch
def make_tool_response(agent: LlmAgent[OllamaClientConfig], message: Message, result):
    m = {"role": "user", "content": f"Output of tool call: {result}"}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[OllamaClientConfig], response: ChatResponse
) -> list:
    return response.message


@dispatch
def extract_answer(
    agent: LlmAgent[OllamaClientConfig],
    extractor: Callable,
    response: ChatResponse,
):
    return extract_answer(agent, extractor, response.message)


@dispatch
def extract_answer(
    agent: LlmAgent[OllamaClientConfig],
    extractor: Callable,
    message: Message,
):
    return extractor(message.content)


@dispatch
def get_text_body(message: Message):
    return message.content


@dispatch
def get_text_body(response: ChatResponse):
    return response.message.content


@dispatch
def get_text_body(tool_call: Message.ToolCall):
    return f"{tool_call.function.name}({tool_call.function.arguments})"


@dispatch
def normalize_message(agent: LlmAgent[OllamaClientConfig], response: ChatResponse):
    return NormalizedMessage(
        kind="assistant_text",
        text=get_text_body(response),
        raw=response,
    )


@dispatch
def normalize_message(agent: LlmAgent[OllamaClientConfig], message: Message):
    return NormalizedMessage(
        kind="assistant_text",
        text=get_text_body(message),
        raw=message,
    )


@dispatch
def normalize_message(agent: LlmAgent[OllamaClientConfig], tool_call: Message.ToolCall):
    return NormalizedMessage(
        kind="tool_call",
        text=get_text_body(tool_call),
        tool_name=tool_call.function.name,
        tool_args=tool_call.function.arguments,
        raw=tool_call,
    )


@dispatch
def count_message_tokens(agent: LlmAgent[OllamaClientConfig], message: dict):
    return count_text_tokens(agent, message["content"])
