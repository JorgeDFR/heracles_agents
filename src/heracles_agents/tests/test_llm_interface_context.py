from types import SimpleNamespace

import pytest
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall

import heracles_agents.llm_interface as llm_interface
from heracles_agents.exceptions import LlmRateLimitError
from heracles_agents.llm_interface import (
    AgentContext,
    AgentResponse,
    AgentSequence,
    AnalyzedExperiment,
    AnalyzedQuestion,
    AnalyzedQuestions,
    CostMetrics,
    LlmCallCost,
    LatencyMetrics,
    QuestionAnalysis,
    EvalQuestion,
    make_cost_metrics,
    make_latency_metrics,
    is_answer_tool_call,
    needs_tool_processing,
    process_answer,
)
from heracles_agents.normalized_response import NormalizedMessage


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
        name="sldp_answer_tool",
        type="custom_tool_call",
    )
    agent = make_agent(
        agent_info=SimpleNamespace(prompt_settings=SimpleNamespace(output_type="SLDP_TOOL"))
    )

    assert process_answer(agent, message) == "<answer>2</answer>"
    assert is_answer_tool_call(agent, message)
    assert not needs_tool_processing(agent, message)

    wrong_tool = message.model_copy(update={"name": "other_tool"})
    assert not is_answer_tool_call(agent, wrong_tool)
    with pytest.raises(ValueError, match="Expected SLDP_TOOL answer tool call"):
        process_answer(agent, wrong_tool)

    pddl_message = message.model_copy(
        update={
            "input": "(visited-place P100)",
            "name": "pddl_answer_tool",
        }
    )
    pddl_agent = make_agent(
        agent_info=SimpleNamespace(prompt_settings=SimpleNamespace(output_type="PDDL_TOOL"))
    )
    assert process_answer(pddl_agent, pddl_message) == "(visited-place P100)"
    assert is_answer_tool_call(pddl_agent, pddl_message)

    with pytest.raises(ValueError, match="Unknown output type"):
        process_answer(
            make_agent(
                agent_info=SimpleNamespace(
                    prompt_settings=SimpleNamespace(output_type="UNKNOWN")
                )
            ),
            message,
        )


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
    token_counts = []

    def flaky_call(model_info, tools, response_format, history):
        calls.append((model_info, tools, response_format, history))
        if len(calls) == 1:
            raise LlmRateLimitError("slow down")
        return "ok"

    agent = make_agent(client=SimpleNamespace(call=flaky_call))
    context = AgentContext(agent)

    monkeypatch.setattr(llm_interface, "generate_prompt_for_agent", lambda prompt, agent: ["hello"])
    monkeypatch.setattr(
        llm_interface,
        "count_message_tokens",
        lambda agent, messages: token_counts.append(messages) or 7,
    )
    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda agent_info: ["tool"])
    monkeypatch.setattr(llm_interface.time, "sleep", lambda seconds: None)
    metric_times = iter([1.0, 1.25, 2.0, 2.5, 3.0, 3.5])
    monkeypatch.setattr(llm_interface.time, "perf_counter", lambda: next(metric_times))

    context.initialize_agent("prompt")
    assert context.history == ["hello"]
    assert context.initial_input_tokens == 0
    assert token_counts == []

    assert context.call_llm(context.history) == "ok"
    assert len(calls) == 2
    assert calls[-1][1:] == (["tool"], "text", ["hello"])
    assert token_counts == [["hello"], "ok"]
    assert context.total_input_tokens == 7
    assert context.llm_call_seconds == 0.75
    assert context.retry_wait_seconds == 0.5


def test_agent_context_call_llm_uses_model_response_format(monkeypatch):
    calls = []

    def call(model_info, tools, response_format, history):
        calls.append(response_format)
        return "ok"

    agent = make_agent(
        model_info=SimpleNamespace(
            model="model",
            temperature=0.1,
            response_format="json",
        ),
        client=SimpleNamespace(call=call),
    )
    context = AgentContext(agent)

    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda agent_info: [])

    assert context.call_llm(["history"]) == "ok"
    assert calls == ["json"]


def test_agent_context_records_openrouter_cost_call(monkeypatch):
    response = SimpleNamespace(
        id="gen-1",
        model="served/model",
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4, cost=0.0002),
    )
    agent = make_agent(
        model_info=SimpleNamespace(model="requested/model", temperature=0.1),
        client=SimpleNamespace(
            client_type="openrouter",
            call=lambda *args: response,
        ),
    )
    context = AgentContext(agent)

    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda agent_info: [])
    monkeypatch.setattr(
        llm_interface,
        "count_message_tokens",
        lambda *_args: pytest.fail("provider usage should avoid local token counting"),
    )

    assert context.call_llm(["history"]) is response
    assert len(context.llm_call_costs) == 1
    assert context.llm_call_costs[0].provider == "openrouter"
    assert context.llm_call_costs[0].model_identifier == "served/model"
    assert context.llm_call_costs[0].input_tokens == 12
    assert context.llm_call_costs[0].output_tokens == 4
    assert context.llm_call_costs[0].request_cost_usd == 0.0002
    assert context.llm_call_costs[0].pricing_source == "openrouter_response_usage"
    assert context.llm_call_costs[0].provider_response_id == "gen-1"
    assert context.llm_call_costs[0].observed_call_seconds is not None
    assert context.llm_call_costs[0].throughput_source == "observed_call_wall_time"
    assert context.total_input_tokens == 12
    assert context.total_output_tokens == 4


def test_agent_context_sums_openrouter_usage_tokens_across_calls(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(
                id="gen-1",
                model="served/model",
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    cost=0.001,
                    prompt_tokens_details=SimpleNamespace(
                        cached_tokens=2,
                        cache_write_tokens=3,
                    ),
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
                ),
            ),
            SimpleNamespace(
                id="gen-2",
                model="served/model",
                usage=SimpleNamespace(
                    prompt_tokens=15,
                    completion_tokens=3,
                    cost=0.002,
                    prompt_tokens_details=SimpleNamespace(
                        cached_tokens=8,
                        cache_write_tokens=0,
                    ),
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
                ),
            ),
        ]
    )
    agent = make_agent(
        client=SimpleNamespace(
            client_type="openrouter",
            call=lambda *_args: next(responses),
        )
    )
    context = AgentContext(agent)
    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda _info: [])
    monkeypatch.setattr(
        llm_interface,
        "count_message_tokens",
        lambda *_args: pytest.fail("provider usage should avoid local token counting"),
    )

    context.call_llm(["initial"])
    context.call_llm(["initial", "tool response"])

    assert context.total_input_tokens == 25
    assert context.total_output_tokens == 5
    assert context.cached_input_tokens == 10
    assert context.reasoning_tokens == 3
    assert context.llm_call_costs[1].cached_input_tokens == 8
    assert context.llm_call_costs[1].reasoning_tokens == 2


def test_agent_context_prefers_ollama_response_token_counts(monkeypatch):
    response = SimpleNamespace(prompt_eval_count=42, eval_count=7)
    agent = make_agent(
        client=SimpleNamespace(
            client_type="ollama",
            call=lambda *_args: response,
        )
    )
    context = AgentContext(agent)
    monkeypatch.setattr(llm_interface, "generate_tools_for_agent", lambda _info: [])
    monkeypatch.setattr(
        llm_interface,
        "count_message_tokens",
        lambda *_args: pytest.fail("Ollama counts should avoid local estimation"),
    )

    assert context.call_llm(["history"]) is response
    assert context.total_input_tokens == 42
    assert context.total_output_tokens == 7


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
    monkeypatch.setattr(
        llm_interface,
        "normalize_message",
        lambda agent, message: NormalizedMessage(
            kind="assistant_text",
            text=str(message),
        ),
    )
    monkeypatch.setattr(llm_interface, "needs_tool_processing", lambda agent, message: message == "tool-call")
    monkeypatch.setattr(llm_interface, "call_function", lambda agent, message: "result")
    monkeypatch.setattr(
        llm_interface,
        "make_tool_response",
        lambda agent, message, result: {"message": message, "result": result},
    )
    metric_times = iter([1.0, 1.4])
    monkeypatch.setattr(llm_interface.time, "perf_counter", lambda: next(metric_times))

    assert context.handle_response("response") == [
        {"message": "tool-call", "result": "result"}
    ]
    assert context.n_tool_calls == 1
    assert context.total_output_tokens == 0
    assert context.tool_execution_seconds == pytest.approx(0.4)


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


def test_experiment_result_dump_uses_compact_ordered_yaml_shape():
    question = EvalQuestion(
        uid="q1",
        name="Question",
        question="What is 1 + 1?",
        solution="2",
        correctness_comparator={"comparison_type": "SLDP", "relation": "equal"},
    )
    response = AgentResponse(raw_response="raw", parsed_response="parsed")
    analyzed_question = AnalyzedQuestion(
        question=question,
        answer="2",
        analysis=QuestionAnalysis(
            valid_answer_format=True,
            correct=True,
            input_tokens=10,
            output_tokens=2,
            n_tool_calls=1,
            latency=LatencyMetrics(
                end_to_end_seconds=1.0,
                llm_call_seconds=0.8,
                tool_execution_seconds=0.1,
                parsing_validation_seconds=0.01,
            ),
        ),
        completed=True,
        sequences=[
            AgentSequence(description="main", responses=[response]),
        ],
    )
    experiment = AnalyzedExperiment(
        metadata={"source_experiment": "experiment.yaml"},
        experiment_configurations={
            "configuration": AnalyzedQuestions(analyzed_questions=[analyzed_question])
        },
    )

    dumped_response = response.model_dump(mode="json")
    dumped_question = analyzed_question.model_dump(mode="json")
    dumped_experiment = experiment.model_dump(mode="json")

    assert "analysis" not in dumped_response
    assert list(dumped_question) == [
        "question",
        "answer",
        "analysis",
        "completed",
        "sequences",
    ]
    assert list(dumped_experiment) == ["metadata", "experiment_configurations"]
    dumped_configuration = dumped_experiment["experiment_configurations"][
        "configuration"
    ]
    assert list(dumped_configuration) == ["analysis_summary", "analyzed_questions"]
    assert dumped_configuration["analysis_summary"]["questions"] == 1
    assert dumped_configuration["analysis_summary"]["correct_count"] == 1
    assert (
        dumped_configuration["analysis_summary"]["latency"][
            "end_to_end_seconds_avg"
        ]
        == 1.0
    )
    assert (
        "analysis"
        not in dumped_configuration["analyzed_questions"][0]["sequences"][0][
            "responses"
        ][0]
    )
    assert dumped_question["analysis"]["latency"] == {
        "end_to_end_seconds": 1.0,
        "llm_call_seconds": 0.8,
        "tool_execution_seconds": 0.1,
        "neo4j_query_seconds": 0.0,
        "parsing_validation_seconds": 0.01,
        "retry_wait_seconds": 0.0,
        }
    assert "cost" not in dumped_question["analysis"]
    assert (
        "cost_summary"
        not in dumped_configuration
    )
    assert "local_resources" not in dumped_configuration


def test_make_latency_metrics_aggregates_contexts():
    contexts = [
        SimpleNamespace(
            llm_call_seconds=1.2345678,
            tool_execution_seconds=0.2,
            retry_wait_seconds=0.3,
            local_llm_runtime_metrics=[],
        ),
        SimpleNamespace(
            llm_call_seconds=2.0,
            tool_execution_seconds=0.4,
            retry_wait_seconds=0.0,
            local_llm_runtime_metrics=[],
        ),
    ]

    latency = make_latency_metrics(
        contexts,
        end_to_end_seconds=4.567891,
        parsing_validation_seconds=0.05,
        neo4j_query_seconds=0.25,
    )

    assert latency.end_to_end_seconds == 4.567891
    assert latency.llm_call_seconds == 3.234568
    assert latency.tool_execution_seconds == 0.6
    assert latency.retry_wait_seconds == 0.3
    assert latency.parsing_validation_seconds == 0.05
    assert latency.neo4j_query_seconds == 0.25
    assert latency.load_duration_seconds is None


def test_make_latency_metrics_prefers_ollama_response_timings():
    context = SimpleNamespace(
        llm_call_seconds=12.5,
        successful_llm_call_seconds=12.5,
        tool_execution_seconds=0.0,
        retry_wait_seconds=0.0,
        total_output_tokens=40,
        local_llm_runtime_metrics=[
            {
                "total_duration_seconds": 10.0,
                "load_duration_seconds": 1.0,
                "prompt_eval_count": 100,
                "prompt_eval_duration_seconds": 2.0,
                "eval_count": 40,
                "eval_duration_seconds": 4.0,
                "prompt_tokens_per_second": 50.0,
                "output_tokens_per_second": 10.0,
                "steady_state_seconds": 6.0,
            }
        ],
    )

    latency = make_latency_metrics(contexts=[context], end_to_end_seconds=13.0)

    assert latency.ollama_total_duration_seconds == 10.0
    assert latency.load_duration_seconds == 1.0
    assert latency.prompt_eval_duration_seconds == 2.0
    assert latency.generation_duration_seconds == 4.0
    assert latency.client_overhead_seconds == 2.5
    assert latency.output_tokens_per_second == 10.0
    assert latency.throughput_source == "ollama_eval_duration"


def test_openrouter_cost_metrics_do_not_fetch_generation_stats():
    generation_requests = []
    call = LlmCallCost(
        provider="openrouter",
        model_identifier="served/model",
        input_tokens=11,
        output_tokens=3,
        request_cost_usd=0.0000123,
        pricing_source="openrouter_response_usage",
        provider_response_id="gen-1",
        billed_input_tokens=11,
        billed_output_tokens=3,
        observed_call_seconds=1.2,
        output_tokens_per_second=2.5,
        throughput_source="observed_call_wall_time",
    )
    context = SimpleNamespace(
        agent=SimpleNamespace(
            client=SimpleNamespace(
                get_generation_stats=lambda generation_id: generation_requests.append(
                    generation_id
                )
            )
        ),
        llm_call_costs=[call],
    )

    cost = make_cost_metrics([context])

    assert cost.total_cost_usd == 0.0000123
    assert cost.cost_basis == "provider_reported"
    assert cost.llm_calls[0].model_identifier == "served/model"
    assert cost.llm_calls[0].input_tokens == 11
    assert cost.llm_calls[0].output_tokens == 3
    assert cost.llm_calls[0].billed_input_tokens == 11
    assert cost.llm_calls[0].billed_output_tokens == 3
    assert cost.llm_calls[0].pricing_source == "openrouter_response_usage"
    assert cost.llm_calls[0].observed_call_seconds == 1.2
    assert cost.llm_calls[0].output_tokens_per_second == 2.5
    assert generation_requests == []


def test_openrouter_cost_metrics_fall_back_to_model_pricing():
    call = LlmCallCost(
        provider="openrouter",
        model_identifier="served/model",
        input_tokens=10,
        output_tokens=2,
        pricing_source="openrouter_generation_api_pending",
        provider_response_id="gen-1",
    )
    context = SimpleNamespace(
        agent=SimpleNamespace(
            client=SimpleNamespace(
                get_generation_stats=lambda generation_id: None,
                get_model_pricing=lambda model_identifier: {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                    "request": "0.00001",
                },
            )
        ),
        llm_call_costs=[call],
    )

    cost = make_cost_metrics([context])

    assert cost.total_cost_usd == 0.000024
    assert cost.cost_basis == "provider_pricing_estimated"
    assert cost.llm_calls[0].request_cost_usd == 0.000024
    assert cost.llm_calls[0].pricing_source == "openrouter_model_pricing_api"


def test_analyzed_questions_cost_summary_is_openrouter_only():
    question = EvalQuestion(
        uid="q1",
        name="Question",
        question="What is 1 + 1?",
        solution="2",
        correctness_comparator={"comparison_type": "SLDP", "relation": "equal"},
    )
    analyzed_questions = AnalyzedQuestions(
        analyzed_questions=[
            AnalyzedQuestion(
                question=question,
                answer="2",
                analysis=QuestionAnalysis(
                    valid_answer_format=True,
                    correct=True,
                    input_tokens=10,
                    output_tokens=2,
                    n_tool_calls=0,
                    cost=CostMetrics(
                        total_cost_usd=0.02,
                        cost_basis="provider_reported",
                        llm_calls=[],
                    ),
                ),
                completed=True,
                sequences=[],
            ),
            AnalyzedQuestion(
                question=question.model_copy(update={"uid": "q2"}),
                answer="3",
                analysis=QuestionAnalysis(
                    valid_answer_format=True,
                    correct=False,
                    input_tokens=10,
                    output_tokens=2,
                    n_tool_calls=0,
                    cost=CostMetrics(
                        total_cost_usd=0.03,
                        cost_basis="provider_reported",
                        llm_calls=[],
                    ),
                ),
                completed=True,
                sequences=[],
            ),
        ]
    )

    assert analyzed_questions.cost_summary.total_cost_usd == 0.05
    assert analyzed_questions.cost_summary.cost_per_question_usd == 0.025
    assert analyzed_questions.cost_summary.cost_per_correct_answer_usd == 0.05


def test_agent_context_run_stops_when_step_done_and_processes_answer(monkeypatch):
    agent = make_agent(agent_info=SimpleNamespace(max_iterations=3, prompt_settings=SimpleNamespace(output_type="SLDP")))
    context = AgentContext(agent)
    context.history = ["history"]
    step_results = iter([(True, False), (True, True)])

    monkeypatch.setattr(context, "step", lambda: next(step_results))
    monkeypatch.setattr(llm_interface, "process_answer", lambda agent, message: "answer")

    assert context.run() == (True, "answer")


def test_agent_context_run_returns_no_answer_when_not_done(monkeypatch):
    agent = make_agent(agent_info=SimpleNamespace(max_iterations=2, prompt_settings=SimpleNamespace(output_type="SLDP")))
    context = AgentContext(agent)

    monkeypatch.setattr(context, "step", lambda: (True, False))

    assert context.run() == (False, None)


def test_agent_context_run_returns_no_answer_when_step_fails(monkeypatch):
    agent = make_agent(agent_info=SimpleNamespace(max_iterations=2, prompt_settings=SimpleNamespace(output_type="SLDP")))
    context = AgentContext(agent)

    monkeypatch.setattr(context, "step", lambda: (False, False))

    assert context.run() == (False, None)
