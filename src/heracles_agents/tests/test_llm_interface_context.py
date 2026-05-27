from types import SimpleNamespace

import pytest
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall

import heracles_agents.llm_interface as llm_interface
from heracles_agents.exceptions import LlmRateLimitError
from heracles_agents.llm_interface import (
    AgentContext,
    AgentResponse,
    is_answer_tool_call,
    needs_tool_processing,
    process_answer,
)


def make_agent(**overrides):
    agent = SimpleNamespace(
        agent_info=SimpleNamespace(
            max_iterations=3,
            prompt_settings=SimpleNamespace(output_type="SLDP"),
            tool_interface="custom",
            tools={},
        ),
        model_info=SimpleNamespace(model="model", temperature=0.1),
        client=SimpleNamespace(call=lambda *args: {"response": "ok"}),
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def test_process_answer_uses_custom_tool_input_for_structured_output(monkeypatch):
    message = ResponseCustomToolCall(
        id="id",
        call_id="call",
        input="<answer>2</answer>",
        name="answer",
        type="custom_tool_call",
    )
    agent = make_agent(
        agent_info=SimpleNamespace(prompt_settings=SimpleNamespace(output_type="SLDP_TOOL"))
    )

    assert process_answer(agent, message) == "<answer>2</answer>"
    assert is_answer_tool_call(agent, message)
    assert not needs_tool_processing(agent, message)


def test_process_answer_falls_back_to_extractor(monkeypatch):
    monkeypatch.setattr(
        llm_interface,
        "extract_answer",
        lambda agent, extractor, message: extractor(message),
    )
    agent = make_agent()

    assert process_answer(agent, "prefix <answer>final</answer>") == "final"


def test_agent_context_initialize_and_call_llm_retries(monkeypatch):
    calls = []

    def flaky_call(model_info, tools, response_format, history):
        calls.append((model_info, tools, response_format, history))
        if len(calls) == 1:
            raise LlmRateLimitError("slow down")
        return "ok"

    agent = make_agent(client=SimpleNamespace(call=flaky_call))
    context = AgentContext(agent)

    monkeypatch.setattr(llm_interface, "generate_prompt_for_agent", lambda prompt, agent: ["hello"])
    monkeypatch.setattr(llm_interface, "count_message_tokens", lambda agent, messages: 7)
    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda agent_info: ["tool"])
    monkeypatch.setattr(llm_interface.time, "sleep", lambda seconds: None)

    context.initialize_agent("prompt")
    assert context.history == ["hello"]
    assert context.initial_input_tokens == 7

    assert context.call_llm(context.history) == "ok"
    assert len(calls) == 2
    assert calls[-1][1:] == (["tool"], "text", ["hello"])


def test_agent_context_call_llm_raises_after_retries(monkeypatch):
    agent = make_agent(client=SimpleNamespace(call=lambda *args: (_ for _ in ()).throw(LlmRateLimitError("limited"))))
    context = AgentContext(agent)

    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda agent_info: [])
    monkeypatch.setattr(llm_interface.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="LLM call failed after 5 retries"):
        context.call_llm(["history"])


def test_agent_context_handle_response_processes_only_tool_messages(monkeypatch):
    messages = ["text", "tool-call"]
    agent = make_agent()
    context = AgentContext(agent)

    monkeypatch.setattr(llm_interface, "iterate_messages", lambda agent, response: messages)
    monkeypatch.setattr(llm_interface, "count_message_tokens", lambda agent, message: 2)
    monkeypatch.setattr(llm_interface, "get_text_body", lambda message: str(message))
    monkeypatch.setattr(llm_interface, "needs_tool_processing", lambda agent, message: message == "tool-call")
    monkeypatch.setattr(llm_interface, "call_function", lambda agent, message: "result")
    monkeypatch.setattr(
        llm_interface,
        "make_tool_response",
        lambda agent, message, result: {"message": message, "result": result},
    )

    assert context.handle_response("response") == [
        {"message": "tool-call", "result": "result"}
    ]
    assert context.n_tool_calls == 1
    assert context.total_output_tokens == 4


def test_agent_context_history_done_and_responses(monkeypatch):
    agent = make_agent()
    context = AgentContext(agent)
    context.history = ["start"]

    monkeypatch.setattr(llm_interface, "generate_update_for_history", lambda agent, response: ["update"])
    monkeypatch.setattr(llm_interface, "get_summary_text", lambda response: f"summary:{response}")

    context.update_history("response")
    assert context.history == ["start", "update"]

    monkeypatch.setattr(llm_interface, "iterate_messages", lambda agent, response: ["message"])
    monkeypatch.setattr(llm_interface, "is_answer_tool_call", lambda agent, message: False)
    assert context.check_if_done(context.history, "response", []) is True
    assert context.check_if_done(context.history, "response", ["tool-response"]) is False

    monkeypatch.setattr(llm_interface, "is_answer_tool_call", lambda agent, message: True)
    assert context.check_if_done(context.history, "response", ["tool-response"]) is True

    responses = context.get_agent_responses()
    assert responses == [
        AgentResponse(raw_response="start", parsed_response="summary:start"),
        AgentResponse(raw_response="update", parsed_response="summary:update"),
    ]


def test_agent_context_run_stops_when_step_done_and_processes_answer(monkeypatch):
    agent = make_agent(agent_info=SimpleNamespace(max_iterations=3, prompt_settings=SimpleNamespace(output_type="SLDP")))
    context = AgentContext(agent)
    context.history = ["history"]
    step_results = iter([False, True])

    monkeypatch.setattr(context, "step", lambda: next(step_results))
    monkeypatch.setattr(llm_interface, "process_answer", lambda agent, message: "answer")

    assert context.run() == (True, "answer")


def test_agent_context_run_returns_no_answer_when_not_done(monkeypatch):
    agent = make_agent(agent_info=SimpleNamespace(max_iterations=2, prompt_settings=SimpleNamespace(output_type="SLDP")))
    context = AgentContext(agent)

    monkeypatch.setattr(context, "step", lambda: False)

    assert context.run() == (False, None)
