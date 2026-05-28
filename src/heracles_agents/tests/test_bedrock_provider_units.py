from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import heracles_agents.provider_integrations.bedrock.bedrock_agent_integration as bedrock_agent
from heracles_agents.agent_functions import extract_answer_tag
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.bedrock.bedrock_client import BedrockClientConfig
from heracles_agents.tool_interface import FunctionParameter, ToolDescription


def make_bedrock_agent(**agent_info_overrides):
    agent_info = SimpleNamespace(tool_interface="bedrock", tools={})
    for key, value in agent_info_overrides.items():
        setattr(agent_info, key, value)
    return LlmAgent[BedrockClientConfig].model_construct(
        agent_info=agent_info,
        model_info=SimpleNamespace(model="bedrock_claude-3-haiku", temperature=0.2),
        client=SimpleNamespace(),
    )


def test_bedrock_generate_prompt_adds_custom_tool_descriptions():
    tool = ToolDescription(
        name="echo",
        description="Echo",
        parameters=[FunctionParameter("value", str, "Value")],
        function=lambda value: value,
    )
    agent = make_bedrock_agent(tool_interface="custom", tools={"echo": tool})

    prompt = bedrock_agent.generate_prompt_for_agent(
        Prompt(system="sys", novel_instruction="ask"),
        agent,
    )

    assert prompt[0] == {"role": "user", "content": [{"text": "sys"}]}
    assert "The following tools can be used" in prompt[1]["content"][0]["text"]
    assert "Function name: echo" in prompt[1]["content"][0]["text"]


def test_bedrock_iteration_function_detection_and_tool_calls():
    direct_tool = SimpleNamespace(function=lambda x: x + 1)
    custom_tool = SimpleNamespace(function=lambda value: value.upper())
    agent = make_bedrock_agent(tools={"direct": direct_tool, "custom": custom_tool})
    text_message = {"text": "<tool> custom(value='abc') </tool>"}
    tool_message = {
        "toolUse": {
            "toolUseId": "tool-id",
            "name": "direct",
            "input": {"x": 2},
        }
    }
    response = {"output": {"message": {"content": [text_message, tool_message]}}}
    messages = list(bedrock_agent.iterate_messages(agent, response))

    assert [message.block for message in messages] == [text_message, tool_message]
    assert not bedrock_agent.is_function_call(agent, messages[0])
    assert bedrock_agent.is_function_call(agent, messages[1])
    assert bedrock_agent.normalize_message(agent, messages[0]).text == (
        "<tool> custom(value='abc') </tool>"
    )
    normalized_tool = bedrock_agent.normalize_message(agent, messages[1])
    assert normalized_tool.kind == "tool_call"
    assert normalized_tool.tool_name == "direct"
    assert normalized_tool.tool_args == {"x": 2}
    assert normalized_tool.tool_id == "tool-id"
    assert bedrock_agent.call_function(agent, messages[0]) == "ABC"
    assert bedrock_agent.call_function(agent, messages[1]) == 3
    with pytest.raises(NotImplementedError):
        bedrock_agent.call_function(
            agent,
            bedrock_agent.BedrockMessage({"unsupported": True}),
        )


def test_bedrock_tool_responses_update_and_answer_extraction():
    agent = make_bedrock_agent()
    tool_message = {
        "toolUse": {
            "toolUseId": "tool-id",
            "name": "direct",
            "input": {"x": 2},
        }
    }
    response = {"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}}}

    wrapped_tool_message = bedrock_agent.BedrockMessage(tool_message)
    wrapped_text_message = bedrock_agent.BedrockMessage({"text": "custom"})

    assert bedrock_agent.make_tool_response(agent, wrapped_tool_message, 3) == {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "tool-id",
                    "content": [{"text": "3"}],
                }
            }
        ],
    }
    assert bedrock_agent.make_tool_response(agent, wrapped_text_message, "ok") == {
        "role": "user",
        "content": [{"text": "Output of tool call: ok"}],
    }
    assert bedrock_agent.generate_update_for_history(agent, response) == [
        response["output"]["message"]
    ]
    assert (
        bedrock_agent.extract_answer(
            agent,
            extract_answer_tag,
            {
                "content": [
                    {
                        "text": "<answer>42</answer>"
                    }
                ]
            },
        )
        == "42"
    )
    assert bedrock_agent.extract_answer(agent, lambda text: text, {"bad": "shape"}) == ""


def test_bedrock_token_counting_shapes():
    agent = make_bedrock_agent()
    with patch.object(bedrock_agent, "count_text_tokens", side_effect=lambda agent, value: len(str(value))):
        assert bedrock_agent.count_message_tokens(agent, "abc") == 3
        assert bedrock_agent.count_message_tokens(agent, {"text": "abc"}) == 3
        assert bedrock_agent.count_message_tokens(
            agent, {"content": [{"text": "ab"}, {"text": "c"}]}
        ) == 6
        assert bedrock_agent.count_message_tokens(
            agent, {"message": {"content": [{"text": "ab"}, {"text": "cd"}]}}
        ) == 5
        assert bedrock_agent.count_message_tokens(
            agent,
            {
                "toolUseId": "id",
                "name": "lookup",
                "input": {"arg": "value"},
            },
        ) == 14
        with pytest.raises(NotImplementedError):
            bedrock_agent.count_message_tokens(agent, {"unknown": True})


def test_bedrock_client_build_payload_and_response_format_guard():
    client = BedrockClientConfig.model_construct()
    model_info = SimpleNamespace(temperature=0.4)

    assert client._build_payload(
        "model-id",
        model_info,
        [{"toolSpec": {"name": "tool"}}],
        [{"role": "user"}],
    ) == {
        "modelId": "model-id",
        "messages": [{"role": "user"}],
        "inferenceConfig": {"temperature": 0.4},
        "toolConfig": {"tools": [{"toolSpec": {"name": "tool"}}]},
    }
    assert "toolConfig" not in client._build_payload(
        "model-id", model_info, [], [{"role": "user"}]
    )

    with pytest.raises(NotImplementedError):
        client.call(SimpleNamespace(model="bedrock_claude-3-haiku"), [], "json", [])
