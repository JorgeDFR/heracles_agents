# ruff: noqa: F811
import logging
import time
from typing import Any, Literal, Optional, Sequence, Union

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
from heracles_agents.local_metrics.ollama_runtime import (
    extract_ollama_response_metrics,
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
    # Legacy fields retained so older callers and result files remain readable.
    raw_response: str
    parsed_response: Optional[str]
    role: Optional[str] = None
    kind: Optional[str] = None
    content: Any = None
    reasoning: Any = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_id: Optional[str] = None
    tool_calls: Any = None
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
    output_tokens_per_second: Optional[float] = Field(
        default=None, exclude_if=lambda value: value is None
    )


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
    cached_input_tokens: Optional[int] = None
    cache_write_input_tokens: Optional[int] = None
    generation_time_seconds: Optional[float] = None
    observed_call_seconds: Optional[float] = None
    output_tokens_per_second: Optional[float] = None
    throughput_source: Optional[str] = None


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


class ConfigurationAnalysisSummary(BaseModel):
    questions: int = 0
    completed_count: int = 0
    completed_rate: Optional[float] = None
    valid_answer_count: int = 0
    valid_answer_rate: Optional[float] = None
    correct_count: int = 0
    accuracy: Optional[float] = None
    final_answer_match_count: int = 0
    final_answer_match_rate: Optional[float] = None
    cypher_solution_match_count: int = 0
    cypher_solution_match_evaluated: int = 0
    cypher_solution_match_rate: Optional[float] = None
    tool_executable_count: int = 0
    tool_executable_evaluated: int = 0
    tool_executable_rate: Optional[float] = None
    input_tokens_total: int = 0
    input_tokens_avg: Optional[float] = None
    input_tokens_processed_total: int = 0
    input_tokens_processed_avg: Optional[float] = None
    cached_input_tokens_total: int = 0
    cached_input_tokens_avg: Optional[float] = None
    cache_write_input_tokens_total: int = 0
    cache_write_input_tokens_avg: Optional[float] = None
    output_tokens_total: int = 0
    output_tokens_avg: Optional[float] = None
    tool_calls_total: int = 0
    tool_calls_avg: Optional[float] = None
    output_tokens_per_second: Optional[float] = None
    latency: dict = Field(default_factory=dict)


class LocalResourceMetrics(BaseModel):
    measurement_scope: str = "ollama_container"
    ollama_container_name: Optional[str] = None
    sample_interval_seconds: Optional[float] = None
    baseline_seconds: Optional[float] = None
    baseline_adjusted: bool = True
    warmup: dict = Field(default_factory=dict)
    telemetry: dict = Field(default_factory=dict)
    cpu: dict = Field(default_factory=dict)
    ram: dict = Field(default_factory=dict)
    gpu: dict = Field(default_factory=dict)
    ollama: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class QuestionAnalysis(BaseModel):
    # Information that is relevant about evaluating the response quality of the
    # "whole question"

    valid_answer_format: bool
    correct: bool
    final_answer_match: Optional[bool] = None
    cypher_solution_match: Optional[bool] = None
    tool_executable: Optional[bool] = None
    generated_cypher: Optional[str] = None
    cypher_tool_output: Any = None
    cypher_validation_issues: list[str] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    n_tool_calls: int
    input_tokens_processed: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    cost: Optional[CostMetrics] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_resources: Optional[LocalResourceMetrics] = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def populate_compatibility_metrics(self):
        if self.final_answer_match is None:
            self.final_answer_match = self.correct
        if self.input_tokens_processed == 0 and self.input_tokens:
            self.input_tokens_processed = self.input_tokens
        return self


class AnalyzedQuestion(BaseModel):
    question: EvalQuestion
    answer: Optional[str]
    analysis: Optional[QuestionAnalysis]
    completed: bool = True
    sequences: list[AgentSequence]


class AnalyzedQuestions(BaseModel):
    analysis_summary: Optional[ConfigurationAnalysisSummary] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    cost_summary: Optional[CostSummary] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_resources: Optional[LocalResourceMetrics] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    analyzed_questions: list[AnalyzedQuestion]

    @model_validator(mode="after")
    def populate_configuration_summaries(self):
        if self.analysis_summary is None:
            self.analysis_summary = make_configuration_analysis_summary(
                self.analyzed_questions
            )
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
    successful_llm_seconds = sum(
        getattr(context, "successful_llm_call_seconds", 0.0) for context in contexts
    )
    output_tokens = sum(
        getattr(context, "total_output_tokens", 0) for context in contexts
    )
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
        output_tokens_per_second=(
            round(output_tokens / successful_llm_seconds, 6)
            if successful_llm_seconds > 0
            else None
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


def make_input_token_metrics(contexts) -> dict[str, int]:
    """Return logical, processed, and provider cache input-token totals.

    ``input_tokens`` counts each growing conversation prefix once. By contrast,
    ``input_tokens_processed`` is the sum reported/estimated for every provider
    request, so it intentionally includes repeated context and retries.
    """

    logical = sum(
        getattr(
            context,
            "total_input_tokens",
            getattr(context, "initial_input_tokens", 0),
        )
        for context in contexts
    )
    processed = sum(
        getattr(context, "total_input_tokens_processed", 0) for context in contexts
    )
    return {
        "input_tokens": logical,
        "input_tokens_processed": processed or logical,
        "cached_input_tokens": sum(
            getattr(context, "cached_input_tokens", 0) for context in contexts
        ),
        "cache_write_input_tokens": sum(
            getattr(context, "cache_write_input_tokens", 0) for context in contexts
        ),
    }


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


def make_configuration_analysis_summary(
    analyzed_questions: Sequence["AnalyzedQuestion"],
) -> ConfigurationAnalysisSummary:
    n_questions = len(analyzed_questions)
    completed_count = sum(1 for q in analyzed_questions if q.completed is True)
    valid_answer_count = sum(
        1
        for q in analyzed_questions
        if getattr(getattr(q, "analysis", None), "valid_answer_format", False)
        is True
    )
    correct_count = sum(
        1
        for q in analyzed_questions
        if getattr(getattr(q, "analysis", None), "correct", False) is True
    )
    final_answer_match_count = sum(
        1
        for q in analyzed_questions
        if getattr(getattr(q, "analysis", None), "final_answer_match", False) is True
    )
    cypher_solution_values = [
        getattr(getattr(q, "analysis", None), "cypher_solution_match", None)
        for q in analyzed_questions
    ]
    cypher_solution_evaluated = sum(value is not None for value in cypher_solution_values)
    cypher_solution_match_count = sum(value is True for value in cypher_solution_values)
    tool_executable_values = [
        getattr(getattr(q, "analysis", None), "tool_executable", None)
        for q in analyzed_questions
    ]
    tool_executable_evaluated = sum(value is not None for value in tool_executable_values)
    tool_executable_count = sum(value is True for value in tool_executable_values)
    input_tokens = [
        _number_or_zero(getattr(getattr(q, "analysis", None), "input_tokens", 0))
        for q in analyzed_questions
    ]
    input_tokens_processed = [
        _number_or_zero(
            getattr(getattr(q, "analysis", None), "input_tokens_processed", 0)
        )
        for q in analyzed_questions
    ]
    cached_input_tokens = [
        _number_or_zero(
            getattr(getattr(q, "analysis", None), "cached_input_tokens", 0)
        )
        for q in analyzed_questions
    ]
    cache_write_input_tokens = [
        _number_or_zero(
            getattr(getattr(q, "analysis", None), "cache_write_input_tokens", 0)
        )
        for q in analyzed_questions
    ]
    output_tokens = [
        _number_or_zero(getattr(getattr(q, "analysis", None), "output_tokens", 0))
        for q in analyzed_questions
    ]
    tool_calls = [
        _number_or_zero(getattr(getattr(q, "analysis", None), "n_tool_calls", 0))
        for q in analyzed_questions
    ]
    provider_generation_seconds = 0.0
    provider_output_tokens = 0
    for q in analyzed_questions:
        cost = getattr(getattr(q, "analysis", None), "cost", None)
        for call in getattr(cost, "llm_calls", []):
            timed_seconds = (
                call.generation_time_seconds or call.observed_call_seconds
            )
            if timed_seconds and call.output_tokens is not None:
                provider_generation_seconds += timed_seconds
                provider_output_tokens += call.output_tokens
    llm_call_seconds = _latency_values(analyzed_questions, "llm_call_seconds")

    return ConfigurationAnalysisSummary(
        questions=n_questions,
        completed_count=completed_count,
        completed_rate=_rate(completed_count, n_questions),
        valid_answer_count=valid_answer_count,
        valid_answer_rate=_rate(valid_answer_count, n_questions),
        correct_count=correct_count,
        accuracy=_rate(correct_count, n_questions),
        final_answer_match_count=final_answer_match_count,
        final_answer_match_rate=_rate(final_answer_match_count, n_questions),
        cypher_solution_match_count=cypher_solution_match_count,
        cypher_solution_match_evaluated=cypher_solution_evaluated,
        cypher_solution_match_rate=_rate(
            cypher_solution_match_count, cypher_solution_evaluated
        ),
        tool_executable_count=tool_executable_count,
        tool_executable_evaluated=tool_executable_evaluated,
        tool_executable_rate=_rate(tool_executable_count, tool_executable_evaluated),
        input_tokens_total=sum(input_tokens),
        input_tokens_avg=_avg(input_tokens),
        input_tokens_processed_total=sum(input_tokens_processed),
        input_tokens_processed_avg=_avg(input_tokens_processed),
        cached_input_tokens_total=sum(cached_input_tokens),
        cached_input_tokens_avg=_avg(cached_input_tokens),
        cache_write_input_tokens_total=sum(cache_write_input_tokens),
        cache_write_input_tokens_avg=_avg(cache_write_input_tokens),
        output_tokens_total=sum(output_tokens),
        output_tokens_avg=_avg(output_tokens),
        tool_calls_total=sum(tool_calls),
        tool_calls_avg=_avg(tool_calls),
        output_tokens_per_second=(
            round(provider_output_tokens / provider_generation_seconds, 6)
            if provider_generation_seconds > 0
            else round(sum(output_tokens) / sum(llm_call_seconds), 6)
            if sum(llm_call_seconds) > 0
            else None
        ),
        latency={
            "end_to_end_seconds_total": _sum_or_none(
                _latency_values(analyzed_questions, "end_to_end_seconds")
            ),
            "end_to_end_seconds_avg": _avg(
                _latency_values(analyzed_questions, "end_to_end_seconds")
            ),
            "end_to_end_seconds_p50": _percentile(
                _latency_values(analyzed_questions, "end_to_end_seconds"), 0.50
            ),
            "end_to_end_seconds_p95": _percentile(
                _latency_values(analyzed_questions, "end_to_end_seconds"), 0.95
            ),
            "llm_call_seconds_total": _sum_or_none(
                llm_call_seconds
            ),
            "llm_call_seconds_avg": _avg(
                llm_call_seconds
            ),
            "tool_execution_seconds_total": _sum_or_none(
                _latency_values(analyzed_questions, "tool_execution_seconds")
            ),
            "neo4j_query_seconds_total": _sum_or_none(
                _latency_values(analyzed_questions, "neo4j_query_seconds")
            ),
            "parsing_validation_seconds_total": _sum_or_none(
                _latency_values(analyzed_questions, "parsing_validation_seconds")
            ),
            "retry_wait_seconds_total": _sum_or_none(
                _latency_values(analyzed_questions, "retry_wait_seconds")
            ),
        },
    )


def _latency_values(
    analyzed_questions: Sequence["AnalyzedQuestion"], field_name: str
) -> list[float]:
    values = []
    for analyzed_question in analyzed_questions:
        latency = getattr(getattr(analyzed_question, "analysis", None), "latency", None)
        value = getattr(latency, field_name, None)
        if value is not None:
            values.append(float(value))
    return values


def _number_or_zero(value) -> int:
    if value is None:
        return 0
    return int(value)


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 6) if total else None


def _avg(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _sum_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(values), 6)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 6)


def _populate_openrouter_cost(context, call: LlmCallCost) -> LlmCallCost:
    # OpenRouter includes usage and cost in the chat response. Avoid a second
    # per-call /generation request: it can be eventually consistent and adds a
    # large serial delay to model sweeps. Use measured request wall time for
    # throughput and fetch model pricing only when response cost is unavailable.
    if call.request_cost_usd is not None:
        return call
    return _estimate_openrouter_cost_from_model_pricing(context, call)


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


def _extract_cache_input_tokens(usage) -> tuple[int, int]:
    """Read cache metrics across OpenRouter/OpenAI and Anthropic usage shapes."""

    prompt_details = (
        _get_attr_or_key(usage, "prompt_tokens_details")
        or _get_attr_or_key(usage, "input_tokens_details")
    )
    cached = _coerce_int(_get_attr_or_key(prompt_details, "cached_tokens"))
    cache_write = _coerce_int(
        _get_attr_or_key(prompt_details, "cache_write_tokens")
    )
    if cached is None:
        cached = _coerce_int(_get_attr_or_key(usage, "cache_read_input_tokens"))
    if cache_write is None:
        cache_write = _coerce_int(
            _get_attr_or_key(usage, "cache_creation_input_tokens")
        )
    return cached or 0, cache_write or 0


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


def make_agent_response(resp, *, agent=None):
    """Serialize a history item with stable fields for report rendering."""

    if agent is None:
        return AgentResponse(
            raw_response=str(resp), parsed_response=get_summary_text(resp)
        )
    try:
        normalized = normalize_message(agent, resp)
    except Exception:
        normalized = NormalizedMessage(kind="unknown", text=get_summary_text(resp), raw=resp)

    if isinstance(resp, dict):
        payload = resp
    else:
        model_dump = getattr(resp, "model_dump", None)
        payload = model_dump(mode="json") if callable(model_dump) else {}

    role = payload.get("role") if isinstance(payload, dict) else None
    if normalized.kind in {"tool_call", "answer_tool", "reasoning"}:
        role = role or "assistant"
    elif normalized.kind == "tool_result":
        role = role or "tool"

    content = payload.get("content") if isinstance(payload, dict) else None
    if content in (None, ""):
        content = payload.get("output") if isinstance(payload, dict) else None
    if content in (None, ""):
        content = normalized.text

    reasoning = None
    tool_calls = None
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning") or payload.get("thinking")
        tool_calls = payload.get("tool_calls")

    return AgentResponse(
        raw_response=str(resp),
        parsed_response=get_summary_text(resp),
        role=role,
        kind=normalized.kind,
        content=content,
        reasoning=reasoning,
        tool_name=normalized.tool_name,
        tool_args=normalized.tool_args,
        tool_id=normalized.tool_id,
        tool_calls=tool_calls,
    )


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
        self.total_input_tokens = 0
        self.total_input_tokens_processed = 0
        self.cached_input_tokens = 0
        self.cache_write_input_tokens = 0
        self._last_input_context_tokens = 0
        self._last_provider_input_context_tokens = 0
        self.total_output_tokens = 0
        self.llm_call_seconds = 0.0
        self.successful_llm_call_seconds = 0.0
        self.tool_execution_seconds = 0.0
        self.retry_wait_seconds = 0.0
        self.time_to_first_token_seconds = None
        self.llm_call_costs = []
        self.local_llm_runtime_metrics = []
        self.tool_executions = []

    def initialize_agent(self, prompt):
        self.history = generate_prompt_for_agent(prompt, self.agent)
        self.initial_input_tokens = count_message_tokens(self.agent, self.history)

        logger.debug(f"Agent inintialized with: \n{get_summary_text(self.history)}")

    def call_llm(self, history):
        model_info = self.agent.model_info
        logger.debug(f"Calling llm with history: {history}")

        explicit_tools = generate_tools_for_agent(self.agent.agent_info)

        response_format = getattr(model_info, "response_format", "text")

        try:
            input_tokens = count_message_tokens(self.agent, history)
        except Exception:
            # Provider integrations implement this for real LlmAgent objects;
            # custom clients may expose no tokenizer.
            input_tokens = self.initial_input_tokens
        logical_input_delta = max(
            0, input_tokens - self._last_input_context_tokens
        )
        self.total_input_tokens += logical_input_delta
        self._last_input_context_tokens = max(
            self._last_input_context_tokens, input_tokens
        )

        n_retries = 5
        wait_time_s = 60
        last_exception = None
        for idx in range(n_retries):
            self.total_input_tokens_processed += input_tokens
            try:
                llm_call_started = time.perf_counter()
                response = self.agent.client.call(
                    model_info,
                    explicit_tools,
                    response_format,
                    history,
                )
                call_seconds = time.perf_counter() - llm_call_started
                self.llm_call_seconds += call_seconds
                self.successful_llm_call_seconds += call_seconds
                usage_input_tokens = _coerce_int(
                    _get_attr_or_key(getattr(response, "usage", None), "prompt_tokens")
                    or _get_attr_or_key(
                        getattr(response, "usage", None), "input_tokens"
                    )
                    or getattr(response, "prompt_eval_count", None)
                )
                if usage_input_tokens is not None:
                    self.total_input_tokens_processed += (
                        usage_input_tokens - input_tokens
                    )
                    provider_logical_delta = max(
                        0,
                        usage_input_tokens
                        - self._last_provider_input_context_tokens,
                    )
                    self.total_input_tokens += (
                        provider_logical_delta - logical_input_delta
                    )
                    self._last_provider_input_context_tokens = max(
                        self._last_provider_input_context_tokens,
                        usage_input_tokens,
                    )
                usage = getattr(response, "usage", None)
                cached_tokens, cache_write_tokens = _extract_cache_input_tokens(usage)
                self.cached_input_tokens += cached_tokens
                self.cache_write_input_tokens += cache_write_tokens
                self.record_llm_call_cost(response, input_tokens, call_seconds)
                self.record_local_llm_runtime_metrics(response)
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

    def record_llm_call_cost(
        self, response, input_tokens: int, call_seconds: float | None = None
    ):
        provider = getattr(getattr(self.agent, "client", None), "client_type", None)
        if provider != "openrouter":
            return

        usage = getattr(response, "usage", None)
        usage_input_tokens = _coerce_int(_get_attr_or_key(usage, "prompt_tokens"))
        usage_output_tokens = _coerce_int(
            _get_attr_or_key(usage, "completion_tokens")
        )
        usage_cost = _coerce_float(_get_attr_or_key(usage, "cost"))
        cached_tokens, cache_write_tokens = _extract_cache_input_tokens(usage)
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
                else "openrouter_model_pricing_api_pending",
                provider_response_id=getattr(response, "id", None),
                billed_input_tokens=usage_input_tokens,
                billed_output_tokens=usage_output_tokens,
                cached_input_tokens=cached_tokens,
                cache_write_input_tokens=cache_write_tokens,
                observed_call_seconds=call_seconds,
                output_tokens_per_second=(
                    round(output_tokens / call_seconds, 6)
                    if output_tokens and call_seconds and call_seconds > 0
                    else None
                ),
                throughput_source="observed_call_wall_time"
                if call_seconds is not None
                else None,
            )
        )

    def record_local_llm_runtime_metrics(self, response):
        provider = getattr(getattr(self.agent, "client", None), "client_type", None)
        if provider != "ollama":
            return
        self.local_llm_runtime_metrics.append(
            extract_ollama_response_metrics(response)
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
            normalized = normalize_message(self.agent, message)
            try:
                result = call_function(self.agent, message)
            except Exception as ex:
                self.tool_executions.append(
                    {
                        "tool_name": normalized.tool_name,
                        "tool_args": normalized.tool_args or {},
                        "tool_id": normalized.tool_id,
                        "output": None,
                        "rows": None,
                        "executable": False,
                        "error": f"{type(ex).__name__}: {ex}",
                    }
                )
                raise
            finally:
                self.tool_execution_seconds += (
                    time.perf_counter() - tool_execution_started
                )
            logger.debug(f"function_result: {result}")
            self.tool_executions.append(
                {
                    "tool_name": normalized.tool_name,
                    "tool_args": normalized.tool_args or {},
                    "tool_id": normalized.tool_id,
                    "output": str(result),
                    "rows": getattr(result, "rows", None),
                    "executable": getattr(result, "executable", True),
                    "error": getattr(result, "error", None),
                }
            )
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
        structured_agent = self.agent if isinstance(self.agent, LlmAgent) else None
        return [
            make_agent_response(resp, agent=structured_agent) for resp in self.history
        ]

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
