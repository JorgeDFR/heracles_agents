from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openrouter.components import ChatAssistantMessage, ChatChoice, ChatResult, ChatToolCall

import heracles_agents.provider_integrations.openrouter.openrouter_agent_integration as openrouter_agent
from heracles_agents.exceptions import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmConnectionError,
    LlmRateLimitError,
    LlmTimeoutError,
    LlmUnknownError,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.openrouter.openrouter_client import (
    OpenRouterClientConfig,
)


def make_openrouter_agent(**agent_info_overrides):
    agent_info = SimpleNamespace(tool_interface="openrouter", tools={})
    for key, value in agent_info_overrides.items():
        setattr(agent_info, key, value)
    return LlmAgent[OpenRouterClientConfig].model_construct(
        agent_info=agent_info,
        model_info=SimpleNamespace(model="openrouter/model", temperature=0.2),
        client=SimpleNamespace(),
    )


def make_result(message):
    return ChatResult(
        id="id",
        choices=[ChatChoice(finish_reason="stop", index=0, message=message)],
        created=0,
        model="model",
        object="chat.completion",
        system_fingerprint=None,
    )


def test_openrouter_generate_prompt_adds_custom_tool_descriptions():
    tool = SimpleNamespace(to_custom=lambda: "Function name: echo\n")
    agent = make_openrouter_agent(tool_interface="custom", tools={"echo": tool})

    prompt = openrouter_agent.generate_prompt_for_agent(
        Prompt(system="sys", novel_instruction="ask"),
        agent,
    )

    assert prompt[0] == {"role": "user", "content": "sys"}
    assert "The following tools can be used" in prompt[1]["content"]
    assert "Function name: echo" in prompt[1]["content"]


def test_openrouter_iteration_and_text_body():
    tool_call = ChatToolCall(
        id="tool-id",
        type="function",
        function={"name": "direct", "arguments": '{"x": 2}'},
    )
    message = ChatAssistantMessage(
        role="assistant",
        content="hello",
        tool_calls=[tool_call],
    )
    response = make_result(message)
    empty_message = ChatAssistantMessage(role="assistant", content=None)

    agent = make_openrouter_agent()
    assert list(openrouter_agent.iterate_messages(agent, response)) == [tool_call, message]
    assert openrouter_agent.get_text_body(response) == "hello"
    assert openrouter_agent.get_text_body(message) == "hello"
    assert openrouter_agent.get_text_body(empty_message) == ""
    assert openrouter_agent.get_text_body(tool_call) == 'direct({"x": 2})'


def test_openrouter_function_and_custom_tool_calls():
    direct_tool = SimpleNamespace(function=lambda x: x + 1)
    custom_tool = SimpleNamespace(function=lambda value: value.upper())
    agent = make_openrouter_agent(tools={"direct": direct_tool, "custom": custom_tool})
    tool_call = ChatToolCall(
        id="tool-id",
        type="function",
        function={"name": "direct", "arguments": '{"x": 2}'},
    )
    custom_message = ChatAssistantMessage(
        role="assistant",
        content="<tool> custom(value='abc') </tool>",
    )

    assert openrouter_agent.is_function_call(agent, tool_call)
    assert not openrouter_agent.is_function_call(agent, custom_message)
    assert openrouter_agent.call_function(agent, tool_call) == 3
    assert openrouter_agent.call_function(agent, custom_message) == "ABC"
    assert openrouter_agent.make_tool_response(agent, tool_call, 3) == {
        "role": "tool",
        "tool_call_id": "tool-id",
        "name": "direct",
        "content": "3",
    }
    assert openrouter_agent.make_tool_response(agent, custom_message, "ABC") == {
        "role": "user",
        "content": "Output of tool call: ABC",
    }


def test_openrouter_update_answer_and_tokens():
    message = ChatAssistantMessage(role="assistant", content="prefix <answer>42</answer>")
    response = make_result(message)
    agent = make_openrouter_agent()

    assert openrouter_agent.generate_update_for_history(agent, response) == [message]
    assert (
        openrouter_agent.extract_answer(
            agent,
            lambda text: text.split("<answer>")[1].split("</answer>")[0],
            response,
        )
        == "42"
    )
    assert (
        openrouter_agent.extract_answer(
            agent,
            lambda text: text.split("<answer>")[1].split("</answer>")[0],
            message,
        )
        == "42"
    )
    assert openrouter_agent.extract_answer(
        agent, lambda text: text, ChatAssistantMessage(role="assistant", content=None)
    ) == ""

    encoder = Mock()
    encoder.encode.side_effect = lambda value: list(str(value))
    with patch.object(openrouter_agent.tiktoken, "get_encoding", return_value=encoder):
        assert openrouter_agent.count_message_tokens(agent, {"content": "abcd"}) == 4
        assert openrouter_agent.count_message_tokens(agent, {"role": "assistant"}) == 0


def test_openrouter_client_payload_and_error_mapping():
    client = OpenRouterClientConfig.model_construct()
    model_info = SimpleNamespace(
        model="model",
        temperature=0.1,
        seed=123,
        reasoning="medium",
    )

    assert client._build_payload(model_info, ["tool"], ["message"]) == {
        "model": "model",
        "messages": ["message"],
        "tools": ["tool"],
        "temperature": 0.1,
        "seed": 123,
        "reasoning": {"effort": "medium"},
    }

    for message, error_type in [
        ("429 rate limit", LlmRateLimitError),
        ("timeout while waiting", LlmTimeoutError),
        ("unauthorized auth failure", LlmAuthenticationError),
        ("invalid bad request", LlmBadRequestError),
        ("network connection reset", LlmConnectionError),
        ("something else", LlmUnknownError),
    ]:
        with pytest.raises(error_type):
            client._raise_normalized_error(RuntimeError(message))

    with pytest.raises(ValueError, match="not implemented"):
        client.call(model_info, [], "json", [])

    client._client = SimpleNamespace(chat=SimpleNamespace(send=Mock(return_value="ok")))
    assert client.call(model_info, [], "text", []) == "ok"
    client._client.chat.send.assert_called_once()
