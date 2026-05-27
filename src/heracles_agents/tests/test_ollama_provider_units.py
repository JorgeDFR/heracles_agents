from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from ollama import ChatResponse, Message

import heracles_agents.provider_integrations.ollama.ollama_agent_integration as ollama_agent
from heracles_agents.exceptions import LlmUnknownError
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.ollama.ollama_client import OllamaClientConfig


def make_ollama_agent(**agent_info_overrides):
    agent_info = SimpleNamespace(tool_interface="ollama", tools={})
    for key, value in agent_info_overrides.items():
        setattr(agent_info, key, value)
    return LlmAgent[OllamaClientConfig].model_construct(
        agent_info=agent_info,
        model_info=SimpleNamespace(model="llama", temperature=0.2, seed=None),
        client=SimpleNamespace(),
    )


def make_response(message):
    return ChatResponse(model="llama", created_at="now", message=message, done=True)


def test_ollama_generate_prompt_adds_custom_tool_descriptions():
    tool = SimpleNamespace(to_custom=lambda: "Function name: echo\n")
    agent = make_ollama_agent(tool_interface="custom", tools={"echo": tool})

    prompt = ollama_agent.generate_prompt_for_agent(
        Prompt(system="sys", novel_instruction="ask"),
        agent,
    )

    assert prompt[0] == {"role": "user", "content": "sys"}
    assert "The following tools can be used" in prompt[1]["content"]
    assert "Function name: echo" in prompt[1]["content"]


def test_ollama_iteration_and_text_body():
    tool_call = Message.ToolCall(function={"name": "direct", "arguments": {"x": 2}})
    message = Message(role="assistant", content="hello", tool_calls=[tool_call])
    response = make_response(message)
    agent = make_ollama_agent()

    assert list(ollama_agent.iterate_messages(agent, response)) == [tool_call, message]
    assert ollama_agent.get_text_body(message) == "hello"
    assert ollama_agent.get_text_body(response) == "hello"
    assert ollama_agent.get_text_body(tool_call) == "direct({'x': 2})"


def test_ollama_function_and_custom_tool_calls():
    direct_tool = SimpleNamespace(function=lambda x: x + 1)
    custom_tool = SimpleNamespace(function=lambda value: value.upper())
    agent = make_ollama_agent(tools={"direct": direct_tool, "custom": custom_tool})
    tool_call = Message.ToolCall(function={"name": "direct", "arguments": {"x": 2}})
    custom_message = Message(role="assistant", content="<tool> custom(value='abc') </tool>")

    assert ollama_agent.is_function_call(agent, tool_call)
    assert not ollama_agent.is_function_call(agent, custom_message)
    assert ollama_agent.call_function(agent, tool_call) == 3
    assert ollama_agent.call_function(agent, custom_message) == "ABC"
    assert ollama_agent.make_tool_response(agent, tool_call, 3) == {
        "role": "tool",
        "tool_name": "direct",
        "content": "3",
    }
    assert ollama_agent.make_tool_response(agent, custom_message, "ABC") == {
        "role": "user",
        "content": "Output of tool call: ABC",
    }


def test_ollama_update_answer_and_tokens():
    message = Message(role="assistant", content="prefix <answer>42</answer>")
    response = make_response(message)
    agent = make_ollama_agent()

    assert ollama_agent.generate_update_for_history(agent, response) == [message]
    assert (
        ollama_agent.extract_answer(
            agent,
            lambda text: text.split("<answer>")[1].split("</answer>")[0],
            response,
        )
        == "42"
    )
    assert (
        ollama_agent.extract_answer(
            agent,
            lambda text: text.split("<answer>")[1].split("</answer>")[0],
            message,
        )
        == "42"
    )

    encoder = Mock()
    encoder.encode.side_effect = lambda value: list(str(value))
    with patch.object(ollama_agent.tiktoken, "get_encoding", return_value=encoder):
        assert ollama_agent.count_message_tokens(agent, {"content": "abcd"}) == 4


def test_ollama_client_call_builds_options_and_normalizes_errors():
    chat = Mock(return_value="response")
    client = OllamaClientConfig.model_construct()
    client._chat_func = chat
    model_info = SimpleNamespace(model="llama", temperature=0.3, seed=123)

    assert client.call(model_info, ["tool"], "text", ["message"]) == "response"
    chat.assert_called_once_with(
        model="llama",
        messages=["message"],
        tools=["tool"],
        think=False,
        options={"temperature": 0.3, "seed": 123},
    )

    with pytest.raises(NotImplementedError):
        client.call(model_info, [], "json", [])

    client._chat_func = Mock(side_effect=RuntimeError("boom"))
    with pytest.raises(LlmUnknownError, match="boom"):
        client.call(model_info, [], "text", [])
