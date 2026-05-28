# ruff: noqa: F811
import logging
import re
from typing import Any

from lark.exceptions import LarkError
from plum import dispatch

from heracles_agents.tool_calling.custom_parser import lark_parse_tool
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import NormalizedMessage
from heracles_agents.prompt import Prompt
from heracles_agents.token_utils import count_text_tokens
from heracles_agents.tool_calling.rendering import render_custom_tool

logger = logging.getLogger(__name__)


def build_custom_tool_prompt(tools):
    tool_command = """The following tools can be used to help formulate your answer.
To call a tool, respond with the tool name and arguments between the <tool> and </tool> tags (XML-style format).
Example: <tool> tool_name(arg1=1,arg2=2,arg3='3') </tool>
    You can use tool calls multiple times in a conversation, however only a single tool call per message.
"""
    for tool in tools:
        tool_command += render_custom_tool(tool)
    return tool_command


def call_custom_tool_from_string(tools, tool_string):
    try:
        function_call = lark_parse_tool(tool_string)
    except LarkError:
        return "Improperly formatted tool call. Tool call should look like: <tool> a_tool_call(arg1=1, arg2='2') </tool>"

    # I didn't want to deal with escaping logic in Lark, so the parsed string has unparsed escape sequences like \"
    # This should turn \" in to "
    for k, v in function_call.args.items():
        if isinstance(v, str):
            function_call.args[k] = v.encode("utf-8").decode("unicode_escape")

    return get_tool_function(tools, function_call.name)(**function_call.args)


def get_tool_function(tools, name):
    if name not in tools:
        available = ", ".join(sorted(tools)) or "none"
        raise ValueError(f"Unknown tool {name!r}. Available tools: {available}")
    return tools[name].function


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: object):
    raise NotImplementedError(
        f"Cannot generate prompt for client of type {type(agent.client)}."
    )


def extract_tag(tag, string):
    tag = re.escape(tag)
    matches = re.findall(
        rf"<{tag}>((?:(?!<{tag}>)[\s\S])*?)<\/{tag}>",
        string,
        re.MULTILINE,
    )
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        logger.warning(f"Found multiple {tag} tags in string")
        logger.debug(f"String with multiple tags: {string}")
    return matches[-1]


def extract_answer_tag(string):
    return extract_tag("answer", string)


@dispatch
def is_function_call(agent, message):
    raise NotImplementedError(
        f"is_function_call not implemented for agent type {type(agent)}, message type {type(message)}. Message: {message}"
    )


@dispatch
def is_answer_tool_call(agent, message):
    prompt_settings = getattr(getattr(agent, "agent_info", None), "prompt_settings", None)
    return (
        getattr(prompt_settings, "output_type", None) == "SLDP_TOOL"
        and getattr(message, "name", None) == "sldp_answer_tool"
        and hasattr(message, "input")
    )


@dispatch
def get_answer_tool_payload(agent, message):
    if hasattr(message, "input"):
        return message.input
    raise NotImplementedError(
        f"get_answer_tool_payload not implemented for agent type {type(agent)}, message type {type(message)}. Message: {message}"
    )


@dispatch
def get_text_body(message: dict):
    if "text" in message:
        return message["text"]
    elif "content" in message:
        return message["content"]
    else:
        raise NotImplementedError(f"get_text_body not implemented for dict: {message}")


@dispatch
def get_text_body(message: NormalizedMessage):
    return message.text


@dispatch
def normalize_message(agent, message: NormalizedMessage):
    return message


@dispatch
def normalize_message(agent, message: dict):
    if "role" in message and "content" in message:
        content = message["content"]
        if isinstance(content, str):
            return NormalizedMessage(
                kind="assistant_text",
                text=f"{message['role']}: {content}",
                raw=message,
            )
    if "type" in message and message["type"] == "function_call_output":
        return NormalizedMessage(
            kind="tool_result",
            text=str(message.get("output", "")),
            raw=message,
        )
    return NormalizedMessage(kind="unknown", text=str(message), raw=message)


@dispatch
def normalize_message(agent, message):
    return NormalizedMessage(kind="assistant_text", text=get_text_body(message), raw=message)


@dispatch
def get_text_body(message):
    raise NotImplementedError(
        f"get_body not implemented for message type {type(message)}. Message: {message}"
    )


def is_custom_tool_call(agent, message):
    if agent.agent_info.tool_interface != "custom":
        return False
    content = get_text_body(message)
    if content is None:
        return False
    tool_call = extract_tag("tool", content)
    return tool_call is not None


@dispatch
def iterate_messages(agent, messages):
    raise NotImplementedError(
        f"iterate_messages not implemented for agent type {type(agent)}, messages type {type(messages)}"
    )


@dispatch
def call_function(agent, tool_message):
    raise NotImplementedError(
        f"call_function not implemented for agent type {type(agent)}, tool_message type {type(tool_message)}"
    )


@dispatch
def make_tool_response(agent, tool_call_message, result):
    raise NotImplementedError(
        f"make_tool_response not implemented for agent type {type(agent)}, tool_call_message type {type(tool_call_message)}, result type {type(result)}."
    )


# if response is a `list`, then we run this for *any* agent type
@dispatch(precedence=1)
def generate_update_for_history(agent: Any, response: list) -> list:
    return response


@dispatch
def extract_answer(agent, extractor, message):
    raise NotImplementedError(
        f"extract_answer not implemented for agent type {type(agent)}, message type {type(message)}"
    )


@dispatch
def count_message_tokens(agent: LlmAgent, messages: list):
    return sum([count_message_tokens(agent, m) for m in messages])


@dispatch
def count_message_tokens(agent: LlmAgent, message):
    text = get_text_body(message)
    # Some providers include end-of-text markers in returned content.
    text = text.replace("<|endoftext|>", "")
    return count_text_tokens(agent, text)


@dispatch
def count_message_tokens(agent: LlmAgent, message: type(None)):
    return 0


@dispatch
def count_tool_description_tokens(agent: LlmAgent, explicit_tools: dict):
    logger.debug(
        "Using this string representation for computing tool tokens: ",
        str(explicit_tools),
    )
    return count_text_tokens(agent, str(explicit_tools))


@dispatch
def count_tool_description_tokens(agent: LlmAgent, explicit_tools: list):
    return sum([count_tool_description_tokens(agent, t) for t in explicit_tools])
