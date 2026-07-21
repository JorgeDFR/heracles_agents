from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

import heracles_agents.provider_integrations.huggingface.huggingface_agent_integration as hf_agent
from heracles_agents.agent_functions import extract_answer_tag
from heracles_agents.exceptions import (
    LlmBadRequestError,
    LlmConnectionError,
    LlmRateLimitError,
    LlmUnknownError,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import process_answer
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.huggingface.huggingface_client import (
    HuggingFaceClientConfig,
)
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription


def make_huggingface_agent(**agent_info_overrides):
    agent_info = SimpleNamespace(tool_interface="none", tools={})
    for key, value in agent_info_overrides.items():
        setattr(agent_info, key, value)
    return LlmAgent[HuggingFaceClientConfig].model_construct(
        agent_info=agent_info,
        model_info=SimpleNamespace(model="model", temperature=0.2, seed=None),
        client=SimpleNamespace(),
    )


def test_huggingface_generate_prompt_adds_custom_tool_descriptions():
    tool = ToolDescription(
        name="echo",
        description="Echo",
        parameters=[FunctionParameter("value", str, "Value")],
        function=lambda value: value,
    )
    agent = make_huggingface_agent(tool_interface="custom", tools={"echo": tool})

    prompt = hf_agent.generate_prompt_for_agent(
        Prompt(system="sys", novel_instruction="ask"),
        agent,
    )

    assert prompt[0] == {"role": "system", "content": "sys"}
    assert "The following tools can be used" in prompt[1]["content"]
    assert "Function name: echo" in prompt[1]["content"]
    assert prompt[-1] == {"role": "user", "content": "ask"}


def test_huggingface_iteration_answer_and_custom_tool_calls():
    message = {"role": "assistant", "content": "<answer>42</answer>"}
    custom_message = {"role": "assistant", "content": "<tool> echo(value='abc') </tool>"}
    agent = make_huggingface_agent(
        tool_interface="custom",
        tools={"echo": SimpleNamespace(function=lambda value: value.upper())},
    )

    assert list(hf_agent.iterate_messages(agent, [message])) == [message]
    assert not hf_agent.is_function_call(agent, message)
    assert hf_agent.extract_answer(agent, extract_answer_tag, message) == "42"
    assert hf_agent.call_function(agent, custom_message) == "ABC"
    assert hf_agent.make_tool_response(agent, custom_message, "ABC") == {
        "role": "user",
        "content": "Output of tool call: ABC",
    }
    assert hf_agent.normalize_message(agent, message).text == "<answer>42</answer>"


def test_huggingface_server_call_builds_openai_compatible_payload(monkeypatch):
    client = HuggingFaceClientConfig(
        client_type="huggingface",
        host="http://huggingface:8000",
        max_new_tokens=10,
        top_p=0.9,
    )
    post_json = Mock(
        return_value={
            "choices": [
                {"message": {"role": "assistant", "content": "MATCH (n) RETURN n"}}
            ]
        }
    )
    monkeypatch.setattr(client, "_post_json", post_json)

    response = client.call(
        SimpleNamespace(model="served-model", temperature=0.2, seed=123),
        [],
        "text",
        [{"role": "user", "content": "question"}],
    )

    assert response == [{"role": "assistant", "content": "MATCH (n) RETURN n"}]
    post_json.assert_called_once_with(
        "http://huggingface:8000/v1/chat/completions",
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "question"}],
            "temperature": 0.2,
            "max_tokens": 10,
            "stream": False,
            "seed": 123,
            "top_p": 0.9,
        },
    )


def test_huggingface_server_host_accepts_v1_suffix():
    client = HuggingFaceClientConfig(
        client_type="huggingface",
        host="http://huggingface:8000/v1",
    )

    assert client._chat_completions_url() == (
        "http://huggingface:8000/v1/chat/completions"
    )


def test_huggingface_client_rejects_unsupported_modes():
    client = HuggingFaceClientConfig(client_type="huggingface")
    model_info = SimpleNamespace(model="base", temperature=0.2)

    with pytest.raises(NotImplementedError):
        client.call(model_info, [], "json", [])
    with pytest.raises(NotImplementedError):
        client.call(model_info, ["tool"], "text", [])


def test_huggingface_client_normalizes_errors(monkeypatch):
    client = HuggingFaceClientConfig(client_type="huggingface")
    model_info = SimpleNamespace(model="base", temperature=0.2, seed=None)

    monkeypatch.setattr(client, "_post_json", Mock(side_effect=ValueError("bad body")))
    with pytest.raises(LlmBadRequestError, match="bad body"):
        client.call(model_info, [], "text", [])

    monkeypatch.setattr(client, "_post_json", Mock(side_effect=URLError("down")))
    with pytest.raises(LlmConnectionError, match="down"):
        client.call(model_info, [], "text", [])

    monkeypatch.setattr(client, "_post_json", Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(LlmUnknownError, match="boom"):
        client.call(model_info, [], "text", [])


def test_huggingface_client_preserves_normalized_http_errors():
    client = HuggingFaceClientConfig(client_type="huggingface")
    error = HTTPError(
        url="http://huggingface:8000/v1/chat/completions",
        code=429,
        msg="rate limited",
        hdrs={},
        fp=None,
    )

    with pytest.raises(LlmRateLimitError):
        client._raise_http_error(error)


def test_huggingface_client_rejects_unexpected_response():
    client = HuggingFaceClientConfig(client_type="huggingface")

    with pytest.raises(ValueError, match="Unexpected Hugging Face server response"):
        client._extract_message({"bad": "shape"})


def test_extract_answer_dispatch_matches_validated_huggingface_agent():
    agent = LlmAgent(
        client={
            "client_type": "huggingface",
            "host": "http://huggingface:8000",
        },
        model_info={
            "model": "served-model",
            "temperature": 0.2,
        },
        agent_info={
            "prompt_settings": {
                "base_prompt": {
                    "system": "sys",
                    "novel_instruction": "ask",
                },
                "output_type": "SLDP",
            },
            "tools": [],
            "tool_interface": "none",
            "max_iterations": 1,
        },
    )

    assert process_answer(
        agent,
        {"role": "assistant", "content": "prefix <answer>42</answer>"},
    ) == "42"
