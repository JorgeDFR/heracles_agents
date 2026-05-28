# ruff: noqa: F811, F401

import copy
import logging

from collections.abc import Callable
from dataclasses import dataclass
from plum import dispatch

from heracles_agents.agent_functions import (
    build_custom_tool_prompt,
    call_custom_tool_from_string,
    extract_tag,
    get_tool_function,
    get_text_body,
    normalize_message,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import NormalizedMessage
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.bedrock.bedrock_client import (
    BedrockClientConfig,
)
from heracles_agents.provider_integrations.bedrock.prompt_rendering import (
    render_bedrock_prompt,
)
import heracles_agents.provider_integrations.bedrock.tool_rendering  # noqa: F401
import heracles_agents.provider_integrations.bedrock.token_counting  # noqa: F401
from heracles_agents.token_utils import count_text_tokens

logger = logging.getLogger(__name__)


@dataclass
class BedrockMessage:
    block: dict

    def __contains__(self, key):
        return key in self.block

    def __getitem__(self, key):
        return self.block[key]


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[BedrockClientConfig]):
    p = copy.deepcopy(prompt)

    if agent.agent_info.tool_interface == "custom":
        p.tool_description = build_custom_tool_prompt(agent.agent_info.tools.values())
    return render_bedrock_prompt(p)


@dispatch
def iterate_messages(agent: LlmAgent[BedrockClientConfig], response_dict: dict):
    for m in response_dict["output"]["message"]["content"]:
        yield BedrockMessage(m)


@dispatch
def is_function_call(agent: LlmAgent[BedrockClientConfig], message: BedrockMessage):
    return "toolUse" in message.block


@dispatch
def call_function(agent: LlmAgent[BedrockClientConfig], tool_message: BedrockMessage):
    available_tools = agent.agent_info.tools
    if "text" in tool_message.block:
        tool_string = extract_tag("tool", tool_message.block["text"])
        return call_custom_tool_from_string(available_tools, tool_string)
    elif "toolUse" in tool_message.block:
        name = tool_message.block["toolUse"]["name"]
        args = tool_message.block["toolUse"]["input"]
        return get_tool_function(available_tools, name)(**args)
    else:
        raise NotImplementedError(
            f"Don't know how to call function from: {tool_message.block}"
        )


@dispatch
def make_tool_response(
    agent: LlmAgent[BedrockClientConfig],
    tool_call_message: BedrockMessage,
    result,
):
    if "toolUse" in tool_call_message.block:
        if not isinstance(result, str):
            result = str(result)
        m = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_call_message.block["toolUse"]["toolUseId"],
                        "content": [{"text": result}],
                    }
                }
            ],
        }
    else:
        m = {"role": "user", "content": [{"text": f"Output of tool call: {result}"}]}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[BedrockClientConfig], response: dict
) -> list:
    return response["output"]["message"]


@dispatch
def extract_answer(
    agent: LlmAgent[BedrockClientConfig],
    extractor: Callable,
    message: dict,
):
    try:
        return extractor(message["content"][0]["text"])
    except Exception as ex:
        logger.error(str(ex))
        return ""


@dispatch
def normalize_message(agent, message: BedrockMessage):
    if "text" in message.block:
        return NormalizedMessage(
            kind="assistant_text",
            text=message.block["text"],
            raw=message.block,
        )
    if "toolUse" in message.block:
        tool_use = message.block["toolUse"]
        return NormalizedMessage(
            kind="tool_call",
            text=f"{tool_use['name']}({tool_use['input']})",
            tool_name=tool_use["name"],
            tool_args=tool_use["input"],
            tool_id=tool_use.get("toolUseId"),
            raw=message.block,
        )
    return NormalizedMessage(kind="unknown", text=str(message.block), raw=message.block)


@dispatch
def get_text_body(message: BedrockMessage):
    return normalize_message(None, message).text


@dispatch
def get_bedrock_block_summary(message: BedrockMessage):
    return get_text_body(message)


@dispatch
def get_bedrock_block_summary(message: dict):
    return get_text_body(BedrockMessage(message))



@dispatch
def count_message_tokens(agent: LlmAgent[BedrockClientConfig], message: str):
    return count_text_tokens(agent, message)


@dispatch
def count_message_tokens(agent: LlmAgent[BedrockClientConfig], message: dict):
    if "content" in message:
        # when we sent a message
        num_tokens = 3
        for block in message["content"]:
            for key, value in block.items():
                num_tokens += count_message_tokens(agent, value)
                # num_tokens += len(enc.encode(value))
        return num_tokens
    elif "text" in message:
        return count_text_tokens(agent, message["text"])
    elif "message" in message:
        return count_text_tokens(
            agent, " ".join([c["text"] for c in message["message"]["content"]])
        )
    elif "toolUse" in message:
        return count_message_tokens(agent, message["toolUse"])
    if "toolUseId" in message:
        total = count_text_tokens(agent, message["name"])
        for argname, argval in message["input"].items():
            total += count_text_tokens(agent, argname)
            total += count_text_tokens(agent, argval)
        return total
    else:
        raise NotImplementedError("Not sure how to process message: ", message)


@dispatch
def count_message_tokens(agent: LlmAgent[BedrockClientConfig], message: BedrockMessage):
    return count_message_tokens(agent, message.block)
