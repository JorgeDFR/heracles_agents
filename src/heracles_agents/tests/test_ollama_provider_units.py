from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from ollama import ChatResponse, Message, RequestError, ResponseError

import heracles_agents.provider_integrations.ollama.ollama_agent_integration as ollama_agent
from heracles_agents.agent_functions import extract_answer_tag
from heracles_agents.exceptions import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmConnectionError,
    LlmRateLimitError,
    LlmServiceUnavailableError,
    LlmTimeoutError,
    LlmUnknownError,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.ollama.ollama_client import OllamaClientConfig
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription


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
    tool = ToolDescription(
        name="echo",
        description="Echo",
        parameters=[FunctionParameter("value", str, "Value")],
        function=lambda value: value,
    )
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

    normalized_message = ollama_agent.normalize_message(agent, message)
    normalized_tool = ollama_agent.normalize_message(agent, tool_call)
    assert normalized_message.kind == "assistant_text"
    assert normalized_message.text == "hello"
    assert normalized_tool.kind == "tool_call"
    assert normalized_tool.tool_name == "direct"
    assert normalized_tool.tool_args == {"x": 2}


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
    message = Message(role="assistant", content="<answer>42</answer>")
    response = make_response(message)
    agent = make_ollama_agent()

    assert ollama_agent.generate_update_for_history(agent, response) == [message]
    assert (
        ollama_agent.extract_answer(
            agent,
            extract_answer_tag,
            response,
        )
        == "42"
    )
    assert (
        ollama_agent.extract_answer(
            agent,
            extract_answer_tag,
            message,
        )
        == "42"
    )

    with patch.object(ollama_agent, "count_text_tokens", side_effect=lambda agent, value: len(str(value))):
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


def test_ollama_client_normalizes_provider_errors():
    client = OllamaClientConfig.model_construct()
    model_info = SimpleNamespace(model="llama", temperature=0.3, seed=None)

    for error, error_type in [
        (TimeoutError("slow"), LlmTimeoutError),
        (ConnectionError("down"), LlmConnectionError),
        (RequestError("request failed"), LlmConnectionError),
        (ResponseError("limited", 429), LlmRateLimitError),
        (ResponseError("timeout", 408), LlmTimeoutError),
        (ResponseError("unavailable", 503), LlmServiceUnavailableError),
        (ResponseError("auth", 401), LlmAuthenticationError),
        (ResponseError("bad", 400), LlmBadRequestError),
        (ResponseError("odd", 418), LlmUnknownError),
        (ValueError("bad value"), LlmBadRequestError),
    ]:
        client._chat_func = Mock(side_effect=error)
        with pytest.raises(error_type):
            client.call(model_info, [], "text", [])


def test_ollama_client_passes_explicit_keep_alive():
    chat = Mock(return_value="response")
    client = OllamaClientConfig.model_construct(keep_alive=-1)
    client._chat_func = chat
    model_info = SimpleNamespace(model="llama", temperature=0.0, seed=None)

    client.call(model_info, [], "text", [])

    assert chat.call_args.kwargs["keep_alive"] == -1


@pytest.mark.parametrize(
    ("reasoning", "expected_think"),
    [
        ({"mode": "enabled", "effort": "medium"}, "medium"),
        ({"mode": "enabled", "effort": None}, True),
        ({"mode": "disabled", "effort": None}, False),
    ],
)
def test_ollama_client_normalizes_reasoning(reasoning, expected_think):
    chat = Mock(return_value="response")
    client = OllamaClientConfig.model_construct()
    client._chat_func = chat
    model_info = SimpleNamespace(
        model="model",
        temperature=None,
        seed=None,
        reasoning=reasoning,
    )

    client.call(model_info, [], "text", [])

    assert chat.call_args.kwargs["think"] == expected_think
    assert chat.call_args.kwargs["options"] == {}


@pytest.mark.parametrize("mode", ["unsupported", "provider_default"])
def test_ollama_client_omits_uncontrolled_reasoning(mode):
    chat = Mock(return_value="response")
    client = OllamaClientConfig.model_construct()
    client._chat_func = chat
    model_info = SimpleNamespace(
        model="model",
        temperature=0.2,
        seed=123,
        reasoning={"mode": mode, "effort": None},
    )

    client.call(model_info, [], "text", [])

    assert "think" not in chat.call_args.kwargs
