# ruff: noqa: F811

import copy
import json
import logging

from collections.abc import Callable
from plum import dispatch

from openai.types.responses.response import Response
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from heracles_agents.agent_functions import (
    build_custom_tool_prompt,
    call_custom_tool_from_string,
    extract_tag,
    get_tool_function,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import NormalizedMessage
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)
from heracles_agents.provider_integrations.openai.prompt_rendering import (
    render_openai_prompt,
)
import heracles_agents.provider_integrations.openai.tool_rendering  # noqa: F401
import heracles_agents.provider_integrations.openai.token_counting  # noqa: F401
from heracles_agents.token_utils import count_text_tokens

logger = logging.getLogger(__name__)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OpenaiClientConfig]):
    p = copy.deepcopy(prompt)
    if agent.agent_info.tool_interface == "custom":
        p.tool_description = build_custom_tool_prompt(agent.agent_info.tools.values())
    return render_openai_prompt(p)


@dispatch
def iterate_messages(agent: LlmAgent[OpenaiClientConfig], messages: Response):
    for m in messages.output:
        yield m


@dispatch
def is_function_call(agent: LlmAgent[OpenaiClientConfig], message):
    """is_function_call should return true for messages that can be passed to call_function below"""
    return (
        isinstance(message, ResponseFunctionToolCall)
        and message.type == "function_call"
    )


@dispatch
def is_answer_tool_call(
    agent: LlmAgent[OpenaiClientConfig], message: ResponseCustomToolCall
):
    return (
        agent.agent_info.prompt_settings.output_type == "SLDP_TOOL"
        and message.name == "sldp_answer_tool"
    )


@dispatch
def get_answer_tool_payload(
    agent: LlmAgent[OpenaiClientConfig], message: ResponseCustomToolCall
):
    return message.input


@dispatch
def call_function(
    agent: LlmAgent[OpenaiClientConfig], tool_message: ResponseFunctionToolCall
):
    available_tools = agent.agent_info.tools
    name = tool_message.name
    args = json.loads(tool_message.arguments)
    logger.debug(f"Calling function {name} {args}")
    result = get_tool_function(available_tools, name)(**args)
    logger.debug(f"Result {result}")
    return result


@dispatch
def call_function(
    agent: LlmAgent[OpenaiClientConfig], tool_message: ResponseOutputMessage
):
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_message.content[0].text)
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenaiClientConfig],
    tool_call_message: ResponseFunctionToolCall,
    result,
):
    m = {
        "type": "function_call_output",
        "call_id": tool_call_message.call_id,
        "output": str(result),
    }
    return m


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenaiClientConfig],
    tool_call_message: ResponseOutputMessage,
    result,
):
    m = {"role": "user", "content": f"Output of tool call: {result}"}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[OpenaiClientConfig], response: Response
) -> list:
    return response.output


@dispatch
def extract_answer(
    agent: LlmAgent[OpenaiClientConfig],
    extractor: Callable,
    message: ResponseOutputMessage,
):
    return extractor(message.content[0].text)


@dispatch
def get_text_body(response: Response):
    return "\n".join([get_text_body(m) for m in response.output])


@dispatch
def get_text_body(message: ResponseOutputMessage):
    return "\n".join([c.text for c in message.content])


@dispatch
def get_text_body(tool_call: ResponseFunctionToolCall):
    return f"{tool_call.name}({tool_call.arguments})"


@dispatch
def get_text_body(message: ResponseReasoningItem):
    if message.content is None:
        return ""
    return "\n".join(c.text for c in message.content)


@dispatch
def get_text_body(tool_call: ResponseCustomToolCall):
    return f"{tool_call.name}({tool_call.input})"


@dispatch
def normalize_message(agent: LlmAgent[OpenaiClientConfig], response: Response):
    return NormalizedMessage(
        kind="assistant_text",
        text=get_text_body(response),
        raw=response,
    )


@dispatch
def normalize_message(agent: LlmAgent[OpenaiClientConfig], message: ResponseOutputMessage):
    return NormalizedMessage(
        kind="assistant_text",
        text=get_text_body(message),
        raw=message,
    )


@dispatch
def normalize_message(
    agent: LlmAgent[OpenaiClientConfig], tool_call: ResponseFunctionToolCall
):
    args = json.loads(tool_call.arguments)
    return NormalizedMessage(
        kind="tool_call",
        text=get_text_body(tool_call),
        tool_name=tool_call.name,
        tool_args=args,
        tool_id=tool_call.call_id,
        raw=tool_call,
    )


@dispatch
def normalize_message(
    agent: LlmAgent[OpenaiClientConfig], message: ResponseReasoningItem
):
    return NormalizedMessage(
        kind="reasoning",
        text=get_text_body(message),
        raw=message,
    )


@dispatch
def normalize_message(
    agent: LlmAgent[OpenaiClientConfig], tool_call: ResponseCustomToolCall
):
    kind = "answer_tool" if is_answer_tool_call(agent, tool_call) else "tool_call"
    return NormalizedMessage(
        kind=kind,
        text=get_text_body(tool_call),
        tool_name=tool_call.name,
        tool_id=tool_call.call_id,
        raw=tool_call,
    )


@dispatch
def count_message_tokens(agent: LlmAgent[OpenaiClientConfig], message: dict):
    # https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    num_tokens = 3
    for key, value in message.items():
        num_tokens += count_text_tokens(agent, value)
    if key == "name":
        num_tokens += 1
    return num_tokens
