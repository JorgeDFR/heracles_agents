from types import SimpleNamespace
from unittest.mock import Mock, patch

from anthropic.types.message import Message
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock

import heracles_agents.provider_integrations.anthropic.anthropic_agent_integration as anthropic_agent
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.anthropic.anthropic_client import (
    AnthropicClientConfig,
)


def make_anthropic_agent(tools=None):
    return LlmAgent[AnthropicClientConfig].model_construct(
        agent_info=SimpleNamespace(tool_interface="anthropic", tools=tools or {}),
        model_info=SimpleNamespace(model="claude", temperature=0.1),
        client=SimpleNamespace(),
    )


def make_message(content):
    return Message(
        id="msg",
        content=content,
        model="claude",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage={"input_tokens": 1, "output_tokens": 2},
    )


def test_anthropic_prompt_iteration_and_text_body():
    text = TextBlock(text="hello", type="text")
    tool = ToolUseBlock(id="tool-id", input={"x": 1}, name="lookup", type="tool_use")
    message = make_message([text, tool])
    agent = make_anthropic_agent()

    assert anthropic_agent.generate_prompt_for_agent(
        Prompt(system="sys", novel_instruction="ask"),
        agent,
    ) == [
        {"role": "user", "content": "sys"},
        {"role": "user", "content": "ask"},
    ]
    assert list(anthropic_agent.iterate_messages(agent, message)) == [text, tool]
    assert anthropic_agent.get_content_blocks_of_type("text", message) == [text]
    assert anthropic_agent.get_text_body(message) == "hello"
    assert anthropic_agent.get_text_body(text) == "hello"
    assert anthropic_agent.get_text_body(tool) == "lookup({'x': 1})"


def test_anthropic_tool_call_paths():
    direct_tool = SimpleNamespace(function=lambda x: x + 1)
    custom_tool = SimpleNamespace(function=lambda value: value.upper())
    agent = make_anthropic_agent({"direct": direct_tool, "custom": custom_tool})
    tool_call = ToolUseBlock(id="tool-id", input={"x": 2}, name="direct", type="tool_use")
    text_call = TextBlock(text="<tool> custom(value='abc') </tool>", type="text")

    assert anthropic_agent.is_function_call(agent, tool_call)
    assert not anthropic_agent.is_function_call(agent, text_call)
    assert anthropic_agent.call_function(agent, tool_call) == 3
    assert anthropic_agent.call_function(agent, text_call) == "ABC"
    assert anthropic_agent.make_tool_response(agent, tool_call, 3) == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool-id",
                "content": "3",
            }
        ],
    }
    assert anthropic_agent.make_tool_response(agent, text_call, "ABC") == {
        "role": "user",
        "content": "Output of tool call: ABC",
    }


def test_anthropic_update_answer_and_token_counts():
    text = TextBlock(text="prefix <answer>42</answer>", type="text")
    message = make_message([text])
    agent = make_anthropic_agent()

    update = anthropic_agent.generate_update_for_history(agent, message)
    assert len(update) == 1
    assert update[0]["role"] == "assistant"
    assert update[0]["content"] == [text]
    assert anthropic_agent.extract_answer(
        agent,
        lambda body: body.split("<answer>")[1].split("</answer>")[0],
        {"content": [text]},
    ) == "42"

    encoder = Mock()
    encoder.encode.side_effect = lambda value: list(str(value))
    with patch.object(anthropic_agent.tiktoken, "get_encoding", return_value=encoder):
        assert anthropic_agent.count_message_tokens(agent, {"content": "abc"}) == 3
        assert anthropic_agent.count_message_tokens(
            agent, {"content": [{"content": "ab"}]}
        ) == 2
        assert anthropic_agent.count_message_tokens(agent, {"role": "user"}) == 8
        assert anthropic_agent.count_message_tokens(agent, "abcd") == 4
