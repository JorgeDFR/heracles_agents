# ruff: noqa: F811
import logging
import time
from typing import Literal, Optional, Union

from plum import dispatch
from pydantic import BaseModel, Field, model_validator

from heracles_agents.agent_functions import (
    call_function,
    count_message_tokens,
    extract_answer,
    extract_answer_tag,
    get_answer_tool_payload,
    generate_prompt_for_agent,
    generate_update_for_history,
    get_text_body,
    is_answer_tool_call,
    is_custom_tool_call,
    is_function_call,
    iterate_messages,
    make_tool_response,
    normalize_message,
)
from heracles_agents.exceptions import (
    LlmRateLimitError,
    LlmTimeoutError,
    LlmServiceUnavailableError,
    LlmConnectionError,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import (
    NormalizedMessage,
    normalized_summary,
)
from heracles_agents.tool_calling.rendering import has_tool_renderer, render_tool_for_interface

logger = logging.getLogger(__name__)


class SldpComparison(BaseModel):
    comparison_type: Literal["SLDP"]
    relation: str  # equal, subset, superset


class PddlComparison(BaseModel):
    comparison_type: Literal["PDDL"]
    relation: str  # equal, subset, superset


class LLmJudgeComparison(BaseModel):
    comparison_type: Literal["LLM_JUDGE"]


ComparisonType = Union[SldpComparison, PddlComparison, LLmJudgeComparison]


class EvalQuestion(BaseModel):
    name: str
    question: str
    solution: str
    uid: str | int
    tags: Optional[list[str]] = None
    correctness_comparator: ComparisonType = Field(discriminator="comparison_type")


class AgentResponse(BaseModel):
    # "full" response from the LLM (what format?)
    raw_response: str
    # interpretable response from the LLM (e.g., tool call, parsed final answer)
    parsed_response: Optional[str]
    analysis: "ResponseAnalysis" = Field(
        default_factory=lambda: ResponseAnalysis(), exclude=True
    )


class ResponseAnalysis(BaseModel):
    valid_sldp: bool = False
    valid_cypher: bool = False
    tool_call_succeeded: Optional[bool] = None


class AgentSequence(BaseModel):
    # What the "purpose" of this agent sequence is. Eventually should be more
    # structured/dispatchable than string?
    description: str

    responses: list[AgentResponse]


class LatencyMetrics(BaseModel):
    # Per-question wall-clock time and the measurable portions of that time.
    end_to_end_seconds: float = 0.0
    llm_call_seconds: float = 0.0
    tool_execution_seconds: float = 0.0
    neo4j_query_seconds: float = 0.0
    parsing_validation_seconds: float = 0.0
    retry_wait_seconds: float = 0.0
    time_to_first_token_seconds: Optional[float] = None


class LlmCallCost(BaseModel):
    provider: str
    model_identifier: str
    input_tokens: int
    output_tokens: int
    request_cost_usd: Optional[float] = None
    pricing_source: str
    provider_response_id: Optional[str] = None
    billed_input_tokens: Optional[int] = None
    billed_output_tokens: Optional[int] = None


class CostMetrics(BaseModel):
    total_cost_usd: Optional[float] = None
    cost_basis: str
    currency: str = "USD"
    llm_calls: list[LlmCallCost] = Field(default_factory=list)


class CostSummary(BaseModel):
    total_cost_usd: Optional[float] = None
    cost_per_question_usd: Optional[float] = None
    cost_per_correct_answer_usd: Optional[float] = None
    currency: str = "USD"
    cost_basis: str


class QuestionAnalysis(BaseModel):
    # Information that is relevant about evaluating the response quality of the
    # "whole question"

    valid_answer_format: bool
    correct: bool
    input_tokens: int
    output_tokens: int
    n_tool_calls: int
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    cost: Optional[CostMetrics] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class AnalyzedQuestion(BaseModel):
    question: EvalQuestion
    answer: Optional[str]
    analysis: Optional[QuestionAnalysis]
    completed: bool = True
    sequences: list[AgentSequence]


class AnalyzedQuestions(BaseModel):
    analyzed_questions: list[AnalyzedQuestion]
    cost_summary: Optional[CostSummary] = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def populate_cost_summary(self):
        if self.cost_summary is None:
            self.cost_summary = make_cost_summary(self.analyzed_questions)
        return self


class AnalyzedExperiment(BaseModel):
    metadata: dict = Field(default_factory=dict)
    experiment_configurations: dict[str, AnalyzedQuestions]


def make_latency_metrics(
    contexts,
    *,
    end_to_end_seconds: float,
    parsing_validation_seconds: float = 0.0,
    neo4j_query_seconds: float = 0.0,
) -> LatencyMetrics:
    first_token_times = [
        getattr(context, "time_to_first_token_seconds", None)
        for context in contexts
        if getattr(context, "time_to_first_token_seconds", None) is not None
    ]
    return LatencyMetrics(
        end_to_end_seconds=round(end_to_end_seconds, 6),
        llm_call_seconds=round(
            sum(getattr(context, "llm_call_seconds", 0.0) for context in contexts), 6
        ),
        tool_execution_seconds=round(
            sum(
                getattr(context, "tool_execution_seconds", 0.0)
                for context in contexts
            ),
            6,
        ),
        neo4j_query_seconds=round(neo4j_query_seconds, 6),
        parsing_validation_seconds=round(parsing_validation_seconds, 6),
        retry_wait_seconds=round(
            sum(getattr(context, "retry_wait_seconds", 0.0) for context in contexts),
            6,
        ),
        time_to_first_token_seconds=(
            round(min(first_token_times), 6) if first_token_times else None
        ),
    )


def make_cost_metrics(contexts) -> Optional[CostMetrics]:
    calls = []
    for context in contexts:
        for call in getattr(context, "llm_call_costs", []):
            if call.provider != "openrouter":
                continue
            calls.append(_populate_openrouter_cost(context, call))

    if not calls:
        return None

    known_costs = [call.request_cost_usd for call in calls]
    if all(cost is not None for cost in known_costs):
        total_cost_usd = round(sum(known_costs), 12)
        pricing_sources = {call.pricing_source for call in calls}
        if pricing_sources <= {
            "openrouter_response_usage",
            "openrouter_generation_api",
        }:
            cost_basis = "provider_reported"
        elif pricing_sources == {"openrouter_model_pricing_api"}:
            cost_basis = "provider_pricing_estimated"
        else:
            cost_basis = "provider_reported_and_estimated"
    elif any(cost is not None for cost in known_costs):
        total_cost_usd = None
        cost_basis = "provider_reported_partial"
    else:
        total_cost_usd = None
        cost_basis = "provider_reported_unavailable"

    return CostMetrics(
        total_cost_usd=total_cost_usd,
        cost_basis=cost_basis,
        llm_calls=calls,
    )


def make_cost_summary(analyzed_questions: list["AnalyzedQuestion"]):
    question_costs = []
    known_costs = []
    for analyzed_question in analyzed_questions:
        analysis = analyzed_question.analysis
        cost = getattr(analysis, "cost", None)
        if cost is None:
            continue
        question_costs.append(cost.total_cost_usd)
        if cost.total_cost_usd is not None:
            known_costs.append(cost.total_cost_usd)

    if not question_costs:
        return None

    if len(known_costs) != len(question_costs):
        return CostSummary(cost_basis="provider_reported_incomplete")

    total_cost_usd = round(sum(known_costs), 12)
    n_questions = len(analyzed_questions)
    n_correct = sum(
        1
        for analyzed_question in analyzed_questions
        if getattr(getattr(analyzed_question, "analysis", None), "correct", False)
    )
    return CostSummary(
        total_cost_usd=total_cost_usd,
        cost_per_question_usd=round(total_cost_usd / n_questions, 12)
        if n_questions
        else None,
        cost_per_correct_answer_usd=round(total_cost_usd / n_correct, 12)
        if n_correct
        else None,
        cost_basis="provider_reported",
    )


def _populate_openrouter_cost(context, call: LlmCallCost) -> LlmCallCost:
    if call.request_cost_usd is not None:
        return call

    estimated_call = _estimate_openrouter_cost_from_model_pricing(context, call)
    if estimated_call.request_cost_usd is not None:
        return estimated_call

    if call.provider_response_id is None:
        return call

    client = getattr(getattr(context, "agent", None), "client", None)
    get_generation_stats = getattr(client, "get_generation_stats", None)
    if get_generation_stats is None:
        return call

    stats = get_generation_stats(call.provider_response_id)
    if not isinstance(stats, dict):
        return _estimate_openrouter_cost_from_model_pricing(context, call)

    request_cost_usd = _coerce_float(stats.get("total_cost"))
    billed_input_tokens = _coerce_int(
        stats.get("native_tokens_prompt", stats.get("tokens_prompt"))
    )
    billed_output_tokens = _coerce_int(
        stats.get("native_tokens_completion", stats.get("tokens_completion"))
    )
    model_identifier = stats.get("model") or call.model_identifier

    updated_call = call.model_copy(
        update={
            "model_identifier": model_identifier,
            "input_tokens": billed_input_tokens or call.input_tokens,
            "output_tokens": billed_output_tokens or call.output_tokens,
            "request_cost_usd": request_cost_usd,
            "pricing_source": "openrouter_generation_api"
            if request_cost_usd is not None
            else "openrouter_generation_api_unavailable",
            "billed_input_tokens": billed_input_tokens,
            "billed_output_tokens": billed_output_tokens,
        }
    )
    if request_cost_usd is None:
        return _estimate_openrouter_cost_from_model_pricing(context, updated_call)
    return updated_call


def _estimate_openrouter_cost_from_model_pricing(
    context, call: LlmCallCost
) -> LlmCallCost:
    client = getattr(getattr(context, "agent", None), "client", None)
    get_model_pricing = getattr(client, "get_model_pricing", None)
    if get_model_pricing is None:
        return call

    pricing = get_model_pricing(call.model_identifier)
    if not isinstance(pricing, dict):
        return call

    prompt_price = _coerce_float(pricing.get("prompt")) or 0.0
    completion_price = _coerce_float(pricing.get("completion")) or 0.0
    request_price = _coerce_float(pricing.get("request")) or 0.0
    request_cost_usd = (
        call.input_tokens * prompt_price
        + call.output_tokens * completion_price
        + request_price
    )

    return call.model_copy(
        update={
            "request_cost_usd": round(request_cost_usd, 12),
            "pricing_source": "openrouter_model_pricing_api",
        }
    )


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_attr_or_key(value, name):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def generate_tools_for_agent(agent_info):
    if agent_info.tool_interface in {"custom", "none"}:
        return []
    if not has_tool_renderer(agent_info.tool_interface):
        raise NotImplementedError(
            f"Unknown tool interface: {agent_info.tool_interface}"
        )
    return [
        render_tool_for_interface(tool, agent_info.tool_interface)
        for tool in agent_info.tools.values()
    ]


def process_answer(agent: LlmAgent, message):
    output_type = agent.agent_info.prompt_settings.output_type
    match output_type:
        case "SLDP_TOOL" | "PDDL_TOOL":
            if not is_answer_tool_call(agent, message):
                raise ValueError(
                    f"Expected {output_type} answer tool call, got {type(message)}: {message}"
                )
            return get_answer_tool_payload(agent, message)
        case "SLDP" | "PDDL" | None:
            return extract_answer(agent, extract_answer_tag, message)
        case _:
            raise ValueError(f"Unknown output type: {output_type}")


def needs_tool_processing(agent: LlmAgent, message):
    if is_answer_tool_call(agent, message):
        return False
    return is_function_call(agent, message) or is_custom_tool_call(agent, message)


@dispatch
def get_summary_text(resp: str):
    return resp


@dispatch
def get_summary_text(resp: type(None)):
    return ""


@dispatch
def get_summary_text(resp: list):
    return "\n".join(get_summary_text(r) for r in resp)


@dispatch
def get_summary_text(resp):
    return normalized_summary(normalize_message(None, resp))


@dispatch
def get_summary_text(resp: NormalizedMessage):
    return normalized_summary(resp)


def make_agent_response(resp):
    return AgentResponse(raw_response=str(resp), parsed_response=get_summary_text(resp))


@dispatch
def get_summary_text(resp: dict):
    if "text" in resp:
        return resp["text"]
    elif "role" in resp and "content" in resp:
        if isinstance(resp["content"], list):
            return (
                resp["role"]
                + ":"
                + "\n".join(get_summary_text(r) for r in resp["content"])
            )
        else:
            return resp["role"] + ": " + resp["content"]
    elif "type" in resp and resp["type"] == "function_call_output" and "output" in resp:
        return "Function result: " + str(resp["output"])
    elif "toolResult" in resp:
        try:
            return "Tool result: " + "\n".join(
                r["text"] for r in resp["toolResult"]["content"]
            )
        except Exception as ex:
            logger.error("tool result logging format error")
            logger.error(str(ex))
            return "Tool result: " + str(resp["toolResult"])
    else:
        logger.warning(f"Don't know how to turn dict {resp} into elegant string")
        return str(resp)


class AgentContext:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.history = []
        self.n_tool_calls = 0
        self.initial_input_tokens = 0
        self.total_output_tokens = 0
        self.llm_call_seconds = 0.0
        self.tool_execution_seconds = 0.0
        self.retry_wait_seconds = 0.0
        self.time_to_first_token_seconds = None
        self.llm_call_costs = []

    def initialize_agent(self, prompt):
        self.history = generate_prompt_for_agent(prompt, self.agent)
        self.initial_input_tokens = count_message_tokens(self.agent, self.history)

        logger.debug(f"Agent inintialized with: \n{get_summary_text(self.history)}")

    def call_llm(self, history):
        model_info = self.agent.model_info
        logger.debug(f"Calling llm with history: {history}")

        explicit_tools = generate_tools_for_agent(self.agent.agent_info)

        response_format = getattr(model_info, "response_format", "text")

        n_retries = 5
        wait_time_s = 60
        last_exception = None
        for idx in range(n_retries):
            try:
                llm_call_started = time.perf_counter()
                provider = getattr(
                    getattr(self.agent, "client", None), "client_type", None
                )
                input_tokens = (
                    count_message_tokens(self.agent, history)
                    if provider == "openrouter"
                    else 0
                )
                response = self.agent.client.call(
                    model_info,
                    explicit_tools,
                    response_format,
                    history,
                )
                self.llm_call_seconds += time.perf_counter() - llm_call_started
                self.record_llm_call_cost(response, input_tokens)
                return response

            # ----------------------------------------------------------
            # Retryable errors
            # ----------------------------------------------------------
            except (
                LlmRateLimitError,
                LlmTimeoutError,
                LlmServiceUnavailableError,
                LlmConnectionError,
            ) as ex:
                self.llm_call_seconds += time.perf_counter() - llm_call_started
                last_exception = ex
                logging.warning(
                    f"{type(ex).__name__}: {ex}\n"
                    f"Retrying in {wait_time_s}s "
                    f"({idx + 1}/{n_retries})"
                )
                retry_wait_started = time.perf_counter()
                time.sleep(wait_time_s)
                self.retry_wait_seconds += time.perf_counter() - retry_wait_started

            # ----------------------------------------------------------
            # Fatal provider errors
            # ----------------------------------------------------------
            except Exception as ex:
                self.llm_call_seconds += time.perf_counter() - llm_call_started
                logger.error(f"LLM provider call failed: {ex}")
                raise

        raise RuntimeError(f"LLM call failed after {n_retries} retries") from last_exception

    def record_llm_call_cost(self, response, input_tokens: int):
        provider = getattr(getattr(self.agent, "client", None), "client_type", None)
        if provider != "openrouter":
            return

        usage = getattr(response, "usage", None)
        usage_input_tokens = _coerce_int(_get_attr_or_key(usage, "prompt_tokens"))
        usage_output_tokens = _coerce_int(
            _get_attr_or_key(usage, "completion_tokens")
        )
        usage_cost = _coerce_float(_get_attr_or_key(usage, "cost"))
        output_tokens = usage_output_tokens
        if output_tokens is None:
            try:
                output_tokens = count_message_tokens(self.agent, response)
            except Exception:
                output_tokens = 0

        self.llm_call_costs.append(
            LlmCallCost(
                provider="openrouter",
                model_identifier=getattr(
                    response,
                    "model",
                    getattr(getattr(self.agent, "model_info", None), "model", ""),
                ),
                input_tokens=usage_input_tokens or input_tokens,
                output_tokens=output_tokens or 0,
                request_cost_usd=usage_cost,
                pricing_source="openrouter_response_usage"
                if usage_cost is not None
                else "openrouter_generation_api_pending",
                provider_response_id=getattr(response, "id", None),
                billed_input_tokens=usage_input_tokens,
                billed_output_tokens=usage_output_tokens,
            )
        )

    def handle_response(self, response):
        executed_tool_calls = []
        logger.debug(f"Handling response: {response}")
        for message in iterate_messages(self.agent, response):
            output_tokens = count_message_tokens(self.agent, message)
            self.total_output_tokens += output_tokens
            message_text = normalize_message(self.agent, message).text
            if message_text is not None:
                logger.debug(f"Processing message ({type(message)}: {message_text}")
            if not needs_tool_processing(self.agent, message):
                continue

            self.n_tool_calls += 1

            tool_execution_started = time.perf_counter()
            try:
                result = call_function(self.agent, message)
            finally:
                self.tool_execution_seconds += (
                    time.perf_counter() - tool_execution_started
                )
            logger.debug(f"function_result: {result}")
            tool_response = make_tool_response(self.agent, message, result)
            logger.debug(f"Tool response: {result}")
            executed_tool_calls.append(tool_response)

        return executed_tool_calls

    def update_history(self, response):
        logger.debug(f"History update: \n{get_summary_text(response)}")
        update = generate_update_for_history(self.agent, response)
        if update is None:
            return
        if isinstance(update, list):
            self.history += update
        else:
            self.history.append(update)

    def check_if_done(self, history, response, last_update):
        # If any of the LLM's response messages called an answer tool, we are done
        for message in iterate_messages(self.agent, response):
            if is_answer_tool_call(self.agent, message):
                return True
        # If the LLM didn't call an answer tool, we are done if the LLM didn't call *any* tools.
        return len(last_update) == 0

    def step(self):
        logger.debug("Agent stepping")
        try:
            response = self.call_llm(self.history)
            logger.debug(f"Got response: {response}")

        except Exception:
            response = {
                "role": "assistant",
                "content": "Unable to contact LLM provider server!",
            }
            self.history.append(response)
            return False, False

        self.update_history(response)
        update = self.handle_response(response)
        logger.debug(f"Tool update: {update}")
        self.update_history(update)
        done = self.check_if_done(self.history, response, update)
        return True, done

    def get_agent_responses(self):
        return [make_agent_response(resp) for resp in self.history]

    def run(self):
        for i in range(self.agent.agent_info.max_iterations):
            success, done = self.step()
            if done or not success:
                break
        if success and done:
            answer = process_answer(self.agent, self.history[-1])
        else:
            answer = None
        logger.debug(f"Agent exiting. Finished before max iteration cap? {done}")
        logger.debug(f"Agent used {self.n_tool_calls} tool calls")
        logger.debug(f"Answer: {answer}")
        return done, answer
