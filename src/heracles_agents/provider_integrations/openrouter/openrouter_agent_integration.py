# ruff: noqa: F811

import copy
import json
import logging
import tiktoken

from plum import dispatch
from typing import Callable

from openrouter.components import ChatResult
from openrouter.components import ChatAssistantMessage
from openrouter.components import ChatToolCall

from heracles_agents.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.openrouter.openrouter_client import (
    OpenRouterClientConfig,
)

logger = logging.getLogger(__name__)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OpenRouterClientConfig]):
    p = copy.deepcopy(prompt)
    if agent.agent_info.tool_interface == "custom":
        tool_command = """The following tools can be used to help formulate your answer.
To call a tool, respond with the tool name and arguments between the <tool> and </tool> tags (XML-style format).
Example: <tool> tool_name(arg1=1,arg2=2,arg3='3') </tool>
You can use tool calls multiple times in a conversation, however only a single tool call per message.
"""
        for tool in agent.agent_info.tools.values():
            d = tool.to_custom()
            tool_command += d
        p.tool_description = tool_command
    return p.to_anthropic_json()


@dispatch
def iterate_messages(agent: LlmAgent[OpenRouterClientConfig], response: ChatResult):
    """Yield tool call objects first, then the assistant message dict."""
    message = response.choices[0].message
    for tc in (message.tool_calls or []):
        yield tc
    yield message


@dispatch
def is_function_call(agent: LlmAgent[OpenRouterClientConfig], tool_call: ChatToolCall):
    return True


@dispatch
def is_function_call(agent: LlmAgent[OpenRouterClientConfig], tool_call):
    return False


@dispatch
def call_function(agent: LlmAgent[OpenRouterClientConfig], tool_call: ChatToolCall):
    available_tools = agent.agent_info.tools
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    return available_tools[name].function(**args)


@dispatch
def call_function(agent: LlmAgent[OpenRouterClientConfig], tool_call: ChatAssistantMessage):
    """Handles custom (XML-tag) tool calls carried in a plain message dict."""
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_call.content)
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenRouterClientConfig],
    tool_call_message: ChatToolCall,
    result,
):
    """Build the tool-role message that the API expects after a function call."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_message.id,
        "name": tool_call_message.function.name,
        "content": str(result),
    }


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenRouterClientConfig],
    message: ChatAssistantMessage,
    result,
):
    """Fallback for custom (XML-tag) tool calls — wrap result as a user message."""
    return {"role": "user", "content": f"Output of tool call: {result}"}


@dispatch
def generate_update_for_history(
    agent: LlmAgent[OpenRouterClientConfig],
    response: ChatResult
) -> list:
    return response.choices[0].message


@dispatch
def extract_answer(
    agent: LlmAgent[OpenRouterClientConfig],
    extractor: Callable,
    response: ChatResult,
):
    return extract_answer(agent, extractor, response.choices[0].message)


@dispatch
def extract_answer(
    agent: LlmAgent[OpenRouterClientConfig],
    extractor: Callable,
    message: ChatAssistantMessage,
):
    if message.content is not None:
        return extractor(message.content)
    # elif message.reasoning is not None:
    #     return extractor(message.reasoning)
    else:
        return ""


@dispatch
def get_text_body(response: ChatResult):
    message = response.choices[0].message
    if message.content is not None:
        return message.content
    # elif message.reasoning is not None:
    #     return message.reasoning
    else:
        return ""


@dispatch
def get_text_body(message: ChatAssistantMessage):
    if message.content is not None:
        return message.content
    # if message.reasoning is not None:
    #     return message.reasoning
    else:
        return ""


@dispatch
def get_text_body(tool_call: ChatToolCall):
    return f"{tool_call.function.name}({tool_call.function.arguments})"


@dispatch
def count_message_tokens(agent: LlmAgent[OpenRouterClientConfig], message: dict):
    enc = tiktoken.get_encoding("cl100k_base")
    content = message.get("content") or ""
    return len(enc.encode(content))