from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openai.types.responses.response import Response
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from pydantic import BaseModel

import heracles_agents.provider_integrations.openai.openai_agent_integration as openai_agent
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.openai.openai_client import OpenaiClientConfig
from heracles_agents.tool_interface import FunctionParameter, ToolDescription


def make_openai_agent(**agent_info_overrides):
    agent_info = SimpleNamespace(tool_interface="openai", tools={})
    for key, value in agent_info_overrides.items():
        setattr(agent_info, key, value)
    return LlmAgent[OpenaiClientConfig].model_construct(
        agent_info=agent_info,
        model_info=SimpleNamespace(model="gpt-4.1"),
        client=SimpleNamespace(),
    )


def make_response(output):
    return Response(
        id="resp",
        created_at=0,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model="model",
        object="response",
        output=output,
        parallel_tool_calls=False,
        temperature=1,
        tool_choice="auto",
        tools=[],
        top_p=1,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text={"format": {"type": "text"}},
        truncation="disabled",
        usage=None,
        user=None,
    )


def make_message(text):
    return ResponseOutputMessage(
        id="msg",
        content=[
            ResponseOutputText(
                annotations=[],
                text=text,
                type="output_text",
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )


def test_openai_generate_prompt_adds_custom_tool_descriptions():
    tool = ToolDescription(
        name="echo",
        description="Echo",
        parameters=[FunctionParameter("value", str, "Value")],
        function=lambda value: value,
    )
    agent = make_openai_agent(tool_interface="custom", tools={"echo": tool})

    prompt = openai_agent.generate_prompt_for_agent(
        Prompt(system="sys", novel_instruction="ask"),
        agent,
    )

    assert prompt[0] == {"role": "developer", "content": "sys"}
    assert "The following tools can be used" in prompt[1]["content"]
    assert "Function name: echo" in prompt[1]["content"]


def test_openai_message_iteration_and_text_extraction():
    message = make_message("hello")
    function_call = ResponseFunctionToolCall(
        arguments='{"x": 1}',
        call_id="call",
        name="tool",
        type="function_call",
        id="fc",
    )
    custom_call = ResponseCustomToolCall(
        id="custom",
        call_id="call",
        input="payload",
        name="custom_tool",
        type="custom_tool_call",
    )
    reasoning = ResponseReasoningItem(id="r", summary=[], type="reasoning")
    response = make_response([message, function_call, custom_call, reasoning])
    agent = make_openai_agent()

    assert list(openai_agent.iterate_messages(agent, response)) == [
        message,
        function_call,
        custom_call,
        reasoning,
    ]
    assert openai_agent.get_text_body(message) == "hello"
    assert openai_agent.get_text_body(function_call) == 'tool({"x": 1})'
    assert openai_agent.get_text_body(custom_call) == "custom_tool(payload)"
    assert openai_agent.get_text_body(reasoning) == ""
    assert openai_agent.get_text_body(response) == "hello\ntool({\"x\": 1})\ncustom_tool(payload)\n"

    normalized_message = openai_agent.normalize_message(agent, message)
    normalized_tool = openai_agent.normalize_message(agent, function_call)
    normalized_reasoning = openai_agent.normalize_message(agent, reasoning)
    assert normalized_message.kind == "assistant_text"
    assert normalized_message.text == "hello"
    assert normalized_tool.kind == "tool_call"
    assert normalized_tool.tool_name == "tool"
    assert normalized_tool.tool_args == {"x": 1}
    assert normalized_tool.tool_id == "call"
    assert normalized_reasoning.kind == "reasoning"


def test_openai_function_and_custom_tool_calls():
    direct_tool = SimpleNamespace(function=lambda x: x + 1)
    custom_tool = SimpleNamespace(function=lambda value: value.upper())
    agent = make_openai_agent(tools={"direct": direct_tool, "custom": custom_tool})

    function_call = ResponseFunctionToolCall(
        arguments='{"x": 2}',
        call_id="call-1",
        name="direct",
        type="function_call",
        id="fc",
    )
    custom_message = make_message("<tool> custom(value='abc') </tool>")

    assert openai_agent.is_function_call(agent, function_call)
    assert not openai_agent.is_function_call(agent, custom_message)
    assert openai_agent.call_function(agent, function_call) == 3
    assert openai_agent.call_function(agent, custom_message) == "ABC"
    assert openai_agent.make_tool_response(agent, function_call, 3) == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "3",
    }
    assert openai_agent.make_tool_response(agent, custom_message, "ABC") == {
        "role": "user",
        "content": "Output of tool call: ABC",
    }


def test_openai_answer_tool_normalizes_as_answer_tool():
    agent = make_openai_agent(
        prompt_settings=SimpleNamespace(output_type="SLDP_TOOL"),
    )
    custom_call = ResponseCustomToolCall(
        id="custom",
        call_id="call",
        input="payload",
        name="sldp_answer_tool",
        type="custom_tool_call",
    )

    normalized = openai_agent.normalize_message(agent, custom_call)

    assert normalized.kind == "answer_tool"
    assert normalized.tool_name == "sldp_answer_tool"
    assert normalized.tool_id == "call"


def test_openai_update_and_answer_extraction():
    message = make_message("prefix <answer>42</answer>")
    response = make_response([message])
    agent = make_openai_agent()

    assert openai_agent.generate_update_for_history(agent, response) == [message]
    assert openai_agent.extract_answer(
        agent,
        lambda text: text.split("<answer>")[1].split("</answer>")[0],
        message,
    ) == "42"


def test_openai_token_count_uses_gpt5_alias():
    encoder = Mock()
    encoder.encode.side_effect = lambda text: list(text)
    agent = make_openai_agent()
    agent.model_info.model = "gpt-5.4-mini"

    with patch.object(openai_agent.tiktoken, "encoding_for_model", return_value=encoder) as by_model:
        assert openai_agent.count_message_tokens(agent, {"role": "user", "content": "hi"}) == 9
        by_model.assert_called_once_with("gpt-5-latest")


class DummyFormat(BaseModel):
    value: str


def test_openai_client_builds_payload_and_response_formats():
    client = OpenaiClientConfig.model_construct(timeout=5)
    model_info = SimpleNamespace(
        model="gpt-5.4-mini",
        temperature=0.2,
        seed=123,
        reasoning="medium",
    )

    payload = client._build_payload(model_info, ["tool"], "json", ["message"])

    assert payload["model"] == "gpt-5.4-mini"
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert payload["tools"] == ["tool"]
    assert payload["input"] == ["message"]
    assert payload["parallel_tool_calls"] is False
    assert payload["seed"] == 123
    assert payload["reasoning"] == {"effort": "medium"}

    custom_format = DummyFormat(value="x")
    assert client._build_response_format("text") == {"format": {"type": "text"}}
    assert client._build_response_format(custom_format) is custom_format
    with pytest.raises(ValueError, match="Unknown response_format"):
        client._build_response_format("xml")
