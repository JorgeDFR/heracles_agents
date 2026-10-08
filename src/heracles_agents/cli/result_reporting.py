"""Shared loading, summarization, and display helpers for experiment results."""

from __future__ import annotations

import ast
import json
import os
import re
import webbrowser
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_FILE_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".pddl",
    ".txt",
    ".yaml",
    ".yml",
}

TERMINAL_QUESTION_COLUMNS = {
    "Topic": "name",
    "Question": "question",
    "Solution": "solution",
    "Answer": "answer",
    "Tool Executable": "tool_executable",
    "Cypher Solution / Grounding Match": "cypher_solution_match",
    "Final Answer Match": "final_answer_match",
    "Input Tokens": "input_tokens",
    "Cached Input Tokens": "cached_input_tokens",
    "Output Tokens": "output_tokens",
    "Reasoning Tokens": "reasoning_tokens",
    "Throughput": "output_tokens_per_second",
    "Model Load Duration": "load_duration_seconds",
    "Tool Calls": "n_tool_calls",
}

TERMINAL_SUMMARY_COLUMNS = {
    "Questions": "questions",
    "Tool Executable": "tool_executable",
    "Cypher Solution / Grounding Match": "cypher_solution_match",
    "Final Answer Match": "final_answer_match",
    "Input Tokens": "input_tokens",
    "Cached Input Tokens": "cached_input_tokens",
    "Output Tokens": "output_tokens",
    "Reasoning Tokens": "reasoning_tokens",
    "Throughput": "output_tokens_per_second",
    "Model Load Duration": "load_duration_seconds",
    "Tool Calls": "n_tool_calls",
}


@dataclass
class ProviderModelRef:
    phase: str
    provider: str | None
    model_identifier: str | None
    request_parameters: dict[str, Any] | None = None
    parameter_capabilities: dict[str, Any] | None = None
    require_parameters: bool | None = None


@dataclass
class ConfigurationSummary:
    questions: int = 0
    completed_count: int = 0
    completed_rate: float | None = None
    valid_answer_count: int = 0
    valid_answer_rate: float | None = None
    correct_count: int = 0
    accuracy: float | None = None
    final_answer_match_count: int = 0
    final_answer_match_rate: float | None = None
    cypher_solution_match_count: int = 0
    cypher_solution_match_evaluated: int = 0
    cypher_solution_match_rate: float | None = None
    tool_executable_count: int = 0
    tool_executable_evaluated: int = 0
    tool_executable_rate: float | None = None
    input_tokens_total: int = 0
    input_tokens_avg: float | None = None
    cached_input_tokens_total: int = 0
    cached_input_tokens_avg: float | None = None
    output_tokens_total: int = 0
    output_tokens_avg: float | None = None
    reasoning_tokens_total: int = 0
    reasoning_tokens_avg: float | None = None
    tool_calls_total: int = 0
    tool_calls_avg: float | None = None
    output_tokens_per_second: float | None = None
    end_to_end_latency_total: float | None = None
    end_to_end_latency_avg: float | None = None
    end_to_end_latency_p50: float | None = None
    end_to_end_latency_p95: float | None = None
    llm_latency_total: float | None = None
    llm_latency_avg: float | None = None
    ollama_total_duration_total: float | None = None
    load_duration_total: float | None = None
    load_duration_avg: float | None = None
    prompt_eval_duration_total: float | None = None
    generation_duration_total: float | None = None
    client_overhead_total: float | None = None
    tool_latency_total: float | None = None
    neo4j_latency_total: float | None = None
    validation_latency_total: float | None = None
    retry_wait_total: float | None = None
    cost_total_usd: float | None = None
    cost_per_question_usd: float | None = None
    cost_per_correct_answer_usd: float | None = None


@dataclass
class QuestionResult:
    source_path: Path
    configuration_name: str
    name: str
    question: str
    solution: str
    answer: str | None
    completed: bool
    valid_answer_format: bool | None
    correct: bool | None
    final_answer_match: bool | None
    cypher_solution_match: bool | None
    tool_executable: bool | None
    generated_cypher: str | None
    cypher_tool_output: Any
    cypher_validation_issues: list[str]
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    n_tool_calls: int | None
    latency: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] | None = None
    local_resources: dict[str, Any] | None = None
    n_sequences: int = 0
    sequences: list[dict[str, Any]] = field(default_factory=list)
    provider_model: str = ""
    question_type: str | None = None
    overlap_class: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ConfigurationResult:
    source_path: Path
    configuration_name: str
    provider_models: list[ProviderModelRef]
    questions: list[QuestionResult]
    summary: ConfigurationSummary
    analysis_summary: dict[str, Any] | None
    cost_summary: dict[str, Any] | None
    local_resources: dict[str, Any] | None


@dataclass
class ResultSource:
    path: Path
    metadata: dict[str, Any]
    configurations: list[ConfigurationResult]


def load_result_file(path: Path) -> ResultSource:
    """Load a result YAML file into normalized display data."""

    expanded_path = path.expanduser()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"File not found: {expanded_path}")

    with expanded_path.open("r", encoding="utf-8") as fo:
        data = yaml.safe_load(fo)

    if not isinstance(data, dict):
        raise ValueError(f"Result YAML must contain a mapping: {expanded_path}")

    return result_source_from_dict(data, expanded_path.resolve())


def load_result_files(paths: Sequence[Path]) -> list[ResultSource]:
    return [load_result_file(path) for path in paths]


def result_source_from_analyzed_questions(
    analyzed_questions: Any,
    *,
    title: str = "results",
    source_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResultSource:
    path = source_path or Path("<memory>")
    data = analyzed_questions.model_dump(mode="json")
    provider_models = _provider_models_for(metadata or {}, title)
    questions = _parse_questions(path, title, data.get("analyzed_questions") or [])
    _attach_provider_model_labels(questions, provider_models)
    return ResultSource(
        path=path,
        metadata=metadata or {},
        configurations=[
            ConfigurationResult(
                source_path=path,
                configuration_name=title,
                provider_models=provider_models,
                questions=questions,
                summary=summarize_configuration_from_data(questions, data),
                analysis_summary=_as_dict_or_none(data.get("analysis_summary")),
                cost_summary=_as_dict_or_none(data.get("cost_summary")),
                local_resources=_as_dict_or_none(data.get("local_resources")),
            )
        ],
    )


def result_source_from_dict(data: dict[str, Any], path: Path) -> ResultSource:
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    if "experiment_configurations" in data:
        raw_configurations = data.get("experiment_configurations") or {}
    elif "analyzed_questions" in data:
        raw_configurations = {"results": data}
    else:
        raise ValueError(
            "Result YAML must contain either 'experiment_configurations' or "
            "'analyzed_questions'."
        )

    if not isinstance(raw_configurations, dict):
        raise ValueError("'experiment_configurations' must be a mapping.")

    configurations = []
    for configuration_name, configuration_data in raw_configurations.items():
        if not isinstance(configuration_data, dict):
            if hasattr(configuration_data, "model_dump"):
                configuration_data = configuration_data.model_dump(mode="json")
            else:
                raise ValueError(
                    f"Configuration '{configuration_name}' must contain a mapping."
                )
        provider_models = _provider_models_for(metadata, configuration_name)
        questions = _parse_questions(
            path,
            configuration_name,
            configuration_data.get("analyzed_questions") or [],
        )
        _attach_provider_model_labels(questions, provider_models)
        configurations.append(
            ConfigurationResult(
                source_path=path,
                configuration_name=configuration_name,
                provider_models=provider_models,
                questions=questions,
                summary=summarize_configuration_from_data(
                    questions,
                    configuration_data,
                ),
                analysis_summary=_as_dict_or_none(
                    configuration_data.get("analysis_summary")
                ),
                cost_summary=_as_dict_or_none(configuration_data.get("cost_summary")),
                local_resources=_as_dict_or_none(
                    configuration_data.get("local_resources")
                ),
            )
        )

    return ResultSource(path=path, metadata=metadata, configurations=configurations)


def summarize_configuration(questions: list[QuestionResult]) -> ConfigurationSummary:
    n_questions = len(questions)
    if n_questions == 0:
        return ConfigurationSummary()

    completed_count = sum(1 for q in questions if q.completed is True)
    valid_answer_count = sum(1 for q in questions if q.valid_answer_format is True)
    correct_count = sum(1 for q in questions if q.correct is True)
    final_answer_match_count = sum(1 for q in questions if q.final_answer_match is True)
    cypher_solution_values = [
        q.cypher_solution_match
        for q in questions
        if q.cypher_solution_match is not None
    ]
    tool_executable_values = [
        q.tool_executable for q in questions if q.tool_executable is not None
    ]
    input_tokens = [_number_or_zero(q.input_tokens) for q in questions]
    cached_input_tokens = [
        _number_or_zero(q.cached_input_tokens) for q in questions
    ]
    output_tokens = [_number_or_zero(q.output_tokens) for q in questions]
    reasoning_tokens = [_number_or_zero(q.reasoning_tokens) for q in questions]
    tool_calls = [_number_or_zero(q.n_tool_calls) for q in questions]

    e2e = _metric_values(questions, "end_to_end_seconds")
    llm = _metric_values(questions, "llm_call_seconds")
    tool = _metric_values(questions, "tool_execution_seconds")
    neo4j = _metric_values(questions, "neo4j_query_seconds")
    validation = _metric_values(questions, "parsing_validation_seconds")
    retry = _metric_values(questions, "retry_wait_seconds")
    ollama_total = _metric_values(questions, "ollama_total_duration_seconds")
    load_duration = _metric_values(questions, "load_duration_seconds")
    prompt_eval_duration = _metric_values(
        questions, "prompt_eval_duration_seconds"
    )
    generation_duration = _metric_values(questions, "generation_duration_seconds")
    client_overhead = _metric_values(questions, "client_overhead_seconds")
    costs = [
        _coerce_float((q.cost or {}).get("total_cost_usd"))
        for q in questions
        if q.cost is not None
    ]
    known_costs = [cost for cost in costs if cost is not None]
    cost_total = round(sum(known_costs), 12) if known_costs else None
    provider_output_tokens = 0
    provider_generation_seconds = 0.0
    for q in questions:
        for call in (q.cost or {}).get("llm_calls", []):
            generation_seconds = _coerce_float(
                call.get("generation_time_seconds")
                or call.get("observed_call_seconds")
            )
            call_output_tokens = _coerce_int(call.get("output_tokens"))
            if generation_seconds and call_output_tokens is not None:
                provider_generation_seconds += generation_seconds
                provider_output_tokens += call_output_tokens
    ollama_output_tokens = sum(
        _number_or_zero(question.output_tokens)
        for question in questions
        if _coerce_float(
            question.latency.get("generation_duration_seconds")
        )
        is not None
    )

    return ConfigurationSummary(
        questions=n_questions,
        completed_count=completed_count,
        completed_rate=completed_count / n_questions,
        valid_answer_count=valid_answer_count,
        valid_answer_rate=valid_answer_count / n_questions,
        correct_count=correct_count,
        accuracy=correct_count / n_questions,
        final_answer_match_count=final_answer_match_count,
        final_answer_match_rate=final_answer_match_count / n_questions,
        cypher_solution_match_count=sum(value is True for value in cypher_solution_values),
        cypher_solution_match_evaluated=len(cypher_solution_values),
        cypher_solution_match_rate=(
            sum(value is True for value in cypher_solution_values)
            / len(cypher_solution_values)
            if cypher_solution_values
            else None
        ),
        tool_executable_count=sum(value is True for value in tool_executable_values),
        tool_executable_evaluated=len(tool_executable_values),
        tool_executable_rate=(
            sum(value is True for value in tool_executable_values)
            / len(tool_executable_values)
            if tool_executable_values
            else None
        ),
        input_tokens_total=int(sum(input_tokens)),
        input_tokens_avg=sum(input_tokens) / n_questions,
        cached_input_tokens_total=int(sum(cached_input_tokens)),
        cached_input_tokens_avg=sum(cached_input_tokens) / n_questions,
        output_tokens_total=int(sum(output_tokens)),
        output_tokens_avg=sum(output_tokens) / n_questions,
        reasoning_tokens_total=int(sum(reasoning_tokens)),
        reasoning_tokens_avg=sum(reasoning_tokens) / n_questions,
        tool_calls_total=int(sum(tool_calls)),
        tool_calls_avg=sum(tool_calls) / n_questions,
        output_tokens_per_second=(
            ollama_output_tokens / sum(generation_duration)
            if generation_duration and sum(generation_duration) > 0
            else provider_output_tokens / provider_generation_seconds
            if provider_generation_seconds > 0
            else sum(output_tokens) / sum(llm)
            if llm and sum(llm) > 0
            else None
        ),
        end_to_end_latency_total=_sum_or_none(e2e),
        end_to_end_latency_avg=_avg_or_none(e2e),
        end_to_end_latency_p50=_percentile(e2e, 50),
        end_to_end_latency_p95=_percentile(e2e, 95),
        llm_latency_total=_sum_or_none(llm),
        llm_latency_avg=_avg_or_none(llm),
        ollama_total_duration_total=_sum_or_none(ollama_total),
        load_duration_total=_sum_or_none(load_duration),
        load_duration_avg=_avg_or_none(load_duration),
        prompt_eval_duration_total=_sum_or_none(prompt_eval_duration),
        generation_duration_total=_sum_or_none(generation_duration),
        client_overhead_total=_sum_or_none(client_overhead),
        tool_latency_total=_sum_or_none(tool),
        neo4j_latency_total=_sum_or_none(neo4j),
        validation_latency_total=_sum_or_none(validation),
        retry_wait_total=_sum_or_none(retry),
        cost_total_usd=cost_total,
        cost_per_question_usd=round(cost_total / n_questions, 12)
        if cost_total is not None
        else None,
        cost_per_correct_answer_usd=round(cost_total / correct_count, 12)
        if cost_total is not None and correct_count
        else None,
    )


def summarize_configuration_from_data(
    questions: list[QuestionResult],
    configuration_data: dict[str, Any],
) -> ConfigurationSummary:
    summary = summarize_configuration(questions)
    analysis_summary = _as_dict_or_none(configuration_data.get("analysis_summary"))
    if analysis_summary:
        latency = _as_dict_or_none(analysis_summary.get("latency")) or {}
        summary.end_to_end_latency_total = _coalesce_float(
            latency.get("end_to_end_seconds_total"),
            summary.end_to_end_latency_total,
        )
        summary.end_to_end_latency_avg = _coalesce_float(
            latency.get("end_to_end_seconds_avg"),
            summary.end_to_end_latency_avg,
        )
        summary.end_to_end_latency_p50 = _coalesce_float(
            latency.get("end_to_end_seconds_p50"),
            summary.end_to_end_latency_p50,
        )
        summary.end_to_end_latency_p95 = _coalesce_float(
            latency.get("end_to_end_seconds_p95"),
            summary.end_to_end_latency_p95,
        )
        summary.llm_latency_total = _coalesce_float(
            latency.get("llm_call_seconds_total"),
            summary.llm_latency_total,
        )
        summary.llm_latency_avg = _coalesce_float(
            latency.get("llm_call_seconds_avg"),
            summary.llm_latency_avg,
        )
        summary.ollama_total_duration_total = _coalesce_float(
            latency.get("ollama_total_duration_seconds_total"),
            summary.ollama_total_duration_total,
        )
        summary.load_duration_total = _coalesce_float(
            latency.get("load_duration_seconds_total"),
            summary.load_duration_total,
        )
        summary.load_duration_avg = _coalesce_float(
            latency.get("load_duration_seconds_avg"),
            summary.load_duration_avg,
        )
        summary.prompt_eval_duration_total = _coalesce_float(
            latency.get("prompt_eval_duration_seconds_total"),
            summary.prompt_eval_duration_total,
        )
        summary.generation_duration_total = _coalesce_float(
            latency.get("generation_duration_seconds_total"),
            summary.generation_duration_total,
        )
        summary.client_overhead_total = _coalesce_float(
            latency.get("client_overhead_seconds_total"),
            summary.client_overhead_total,
        )
        summary.tool_latency_total = _coalesce_float(
            latency.get("tool_execution_seconds_total"),
            summary.tool_latency_total,
        )
        summary.neo4j_latency_total = _coalesce_float(
            latency.get("neo4j_query_seconds_total"),
            summary.neo4j_latency_total,
        )
        summary.validation_latency_total = _coalesce_float(
            latency.get("parsing_validation_seconds_total"),
            summary.validation_latency_total,
        )
        summary.retry_wait_total = _coalesce_float(
            latency.get("retry_wait_seconds_total"),
            summary.retry_wait_total,
        )

    cost_summary = _as_dict_or_none(configuration_data.get("cost_summary"))
    if cost_summary:
        summary.cost_total_usd = _coalesce_float(
            cost_summary.get("total_cost_usd"),
            summary.cost_total_usd,
        )
        summary.cost_per_question_usd = _coalesce_float(
            cost_summary.get("cost_per_question_usd"),
            summary.cost_per_question_usd,
        )
        summary.cost_per_correct_answer_usd = _coalesce_float(
            cost_summary.get("cost_per_correct_answer_usd"),
            summary.cost_per_correct_answer_usd,
        )
    return summary


def render_terminal_summary(
    sources: Sequence[ResultSource],
    *,
    console: Console,
    max_width: int = 96,
    summary_only: bool = False,
    show_sequences: bool = False,
) -> None:
    if not sources:
        console.print("[yellow]No result files loaded.[/yellow]")
        return

    columns = dict(TERMINAL_QUESTION_COLUMNS)
    if show_sequences:
        columns["Sequences"] = "n_sequences"

    for source in sources:
        if len(sources) > 1:
            console.rule(f"[bold blue]{source.path}")

        if not source.configurations:
            console.print("[yellow]No experiment configurations found.[/yellow]")
            continue

        for configuration in source.configurations:
            console.rule(
                f"[bold yellow]Configuration: {configuration.configuration_name}"
            )
            rows = [_terminal_question_row(q) for q in configuration.questions]
            if not summary_only:
                console.print(
                    make_table(
                        "Per-Question Results",
                        rows,
                        columns,
                        max_width=max_width,
                    )
                )
            console.print(
                make_table(
                    "Summary",
                    [_terminal_summary_row(configuration.summary)],
                    TERMINAL_SUMMARY_COLUMNS,
                    max_width=max_width,
                )
            )


def render_html_report(
    sources: Sequence[ResultSource],
    output_path: Path,
    *,
    open_report: bool = False,
) -> Path:
    output = output_path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _html_payload(sources)
    data_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = _HTML_TEMPLATE.replace("__GENERATED_AT__", escape(generated_at)).replace(
        "__REPORT_DATA__", data_json
    )
    output.write_text(html, encoding="utf-8")
    if open_report:
        webbrowser.open(output.resolve().as_uri())
    return output


def colorize(color: str, string: Any) -> str:
    return f"[{color}]{string}[/{color}]"


def format_terminal_value(value: Any, max_width: int) -> str:
    if isinstance(value, bool):
        return colorize("green" if value else "red", value)
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    if max_width > 0 and len(text) > max_width:
        return text[: max_width - 1] + "..."
    return text


def make_table(
    title: str,
    rows: list[dict[str, Any]],
    columns: dict[str, str],
    *,
    max_width: int = 96,
) -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for column_name in columns:
        table.add_column(column_name, overflow="fold")

    if not rows:
        table.add_row(*[""] * len(columns))
        return table

    for row in rows:
        table.add_row(
            *[
                format_terminal_value(row.get(field_name), max_width)
                for field_name in columns.values()
            ]
        )
    return table


def _parse_questions(
    source_path: Path,
    configuration_name: str,
    analyzed_questions: list[dict[str, Any]],
) -> list[QuestionResult]:
    questions = []
    for analyzed_question in analyzed_questions:
        if not isinstance(analyzed_question, dict):
            continue
        question_data = analyzed_question.get("question") or {}
        analysis = analyzed_question.get("analysis") or {}
        latency = analysis.get("latency") or {}
        cost = analysis.get("cost")
        local_resources = analysis.get("local_resources")
        sequences = analyzed_question.get("sequences") or []
        questions.append(
            QuestionResult(
                source_path=source_path,
                configuration_name=configuration_name,
                name=_as_text(question_data.get("name")),
                question_type=_as_optional_text(question_data.get("question_type")),
                overlap_class=_as_optional_text(question_data.get("overlap_class")),
                tags=[str(tag) for tag in (question_data.get("tags") or [])],
                question=_as_text(question_data.get("question")),
                solution=_as_text(question_data.get("solution")),
                answer=_as_optional_text(analyzed_question.get("answer")),
                completed=bool(analyzed_question.get("completed", False)),
                valid_answer_format=_as_optional_bool(
                    analysis.get("valid_answer_format")
                ),
                correct=_as_optional_bool(analysis.get("correct")),
                final_answer_match=_as_optional_bool(
                    analysis.get("final_answer_match", analysis.get("correct"))
                ),
                cypher_solution_match=_as_optional_bool(
                    analysis.get("cypher_solution_match")
                ),
                tool_executable=_as_optional_bool(analysis.get("tool_executable")),
                generated_cypher=_as_optional_text(analysis.get("generated_cypher")),
                cypher_tool_output=analysis.get("cypher_tool_output"),
                cypher_validation_issues=[
                    str(issue)
                    for issue in (analysis.get("cypher_validation_issues") or [])
                ],
                input_tokens=_coerce_int(
                    analysis.get("input_tokens_processed", analysis.get("input_tokens"))
                ),
                cached_input_tokens=_coerce_int(
                    analysis.get("cached_input_tokens")
                ),
                output_tokens=_coerce_int(analysis.get("output_tokens")),
                reasoning_tokens=_coerce_int(analysis.get("reasoning_tokens")),
                n_tool_calls=_coerce_int(analysis.get("n_tool_calls")),
                latency=latency if isinstance(latency, dict) else {},
                cost=cost if isinstance(cost, dict) else None,
                local_resources=local_resources
                if isinstance(local_resources, dict)
                else None,
                n_sequences=len(sequences),
                sequences=sequences if isinstance(sequences, list) else [],
            )
        )
    return questions


def _provider_models_for(metadata: dict[str, Any], configuration_name: str):
    llm_configurations = metadata.get("llm_configurations") or {}
    configuration_data = llm_configurations.get(configuration_name) or {}
    phases = configuration_data.get("phases") or {}
    provider_models = []
    if isinstance(phases, dict):
        for phase_name, phase_data in phases.items():
            if not isinstance(phase_data, dict):
                continue
            provider_models.append(
                ProviderModelRef(
                    phase=str(phase_name),
                    provider=phase_data.get("provider"),
                    model_identifier=phase_data.get("model_identifier"),
                    request_parameters=_as_dict_or_none(
                        phase_data.get("request_parameters")
                    ),
                    parameter_capabilities=_as_dict_or_none(
                        phase_data.get("parameter_capabilities")
                    ),
                    require_parameters=phase_data.get("require_parameters"),
                )
            )
    return provider_models


def _attach_provider_model_labels(
    questions: list[QuestionResult], provider_models: list[ProviderModelRef]
) -> None:
    label = _provider_model_label(provider_models)
    for question in questions:
        question.provider_model = label


def _provider_model_label(provider_models: list[ProviderModelRef]) -> str:
    labels = []
    for ref in provider_models:
        provider = ref.provider or ""
        model = ref.model_identifier or ""
        label = "/".join(part for part in [provider, model] if part)
        if len(provider_models) > 1 and ref.phase:
            label = f"{ref.phase}: {label}" if label else ref.phase
        if label:
            labels.append(label)
    return ", ".join(dict.fromkeys(labels))


def _provider_labels(provider_models: list[ProviderModelRef]) -> list[str]:
    return list(
        dict.fromkeys(
            str(ref.provider)
            for ref in provider_models
            if ref.provider is not None and str(ref.provider)
        )
    )


def _terminal_question_row(question: QuestionResult) -> dict[str, Any]:
    return {
        "name": question.name,
        "question": question.question,
        "solution": question.solution,
        "answer": question.answer,
        "completed": question.completed,
        "valid_answer_format": question.valid_answer_format,
        "correct": question.correct,
        "final_answer_match": question.final_answer_match,
        "cypher_solution_match": question.cypher_solution_match,
        "tool_executable": question.tool_executable,
        "input_tokens": question.input_tokens,
        "cached_input_tokens": question.cached_input_tokens,
        "output_tokens": question.output_tokens,
        "reasoning_tokens": question.reasoning_tokens,
        "output_tokens_per_second": question.latency.get(
            "output_tokens_per_second"
        ),
        "load_duration_seconds": question.latency.get("load_duration_seconds"),
        "n_tool_calls": question.n_tool_calls,
        "n_sequences": question.n_sequences,
    }


def _terminal_summary_row(summary: ConfigurationSummary) -> dict[str, Any]:
    return {
        "questions": summary.questions,
        "completed": _count_rate(summary.completed_count, summary.questions),
        "valid_answer_format": _count_rate(
            summary.valid_answer_count, summary.questions
        ),
        "correct": _count_rate(summary.correct_count, summary.questions),
        "final_answer_match": _count_rate(
            summary.final_answer_match_count, summary.questions
        ),
        "cypher_solution_match": _count_rate(
            summary.cypher_solution_match_count,
            summary.cypher_solution_match_evaluated,
        ),
        "tool_executable": _count_rate(
            summary.tool_executable_count,
            summary.tool_executable_evaluated,
        ),
        "input_tokens": _total_avg(
            summary.input_tokens_total, summary.input_tokens_avg
        ),
        "cached_input_tokens": _total_avg(
            summary.cached_input_tokens_total,
            summary.cached_input_tokens_avg,
        ),
        "output_tokens": _total_avg(
            summary.output_tokens_total, summary.output_tokens_avg
        ),
        "reasoning_tokens": _total_avg(
            summary.reasoning_tokens_total, summary.reasoning_tokens_avg
        ),
        "output_tokens_per_second": (
            f"{summary.output_tokens_per_second:.2f} tok/s"
            if summary.output_tokens_per_second is not None
            else None
        ),
        "load_duration_seconds": _total_avg(
            summary.load_duration_total, summary.load_duration_avg
        ),
        "n_tool_calls": _total_avg(summary.tool_calls_total, summary.tool_calls_avg),
    }


def _count_rate(count: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{count}/{total} ({100 * count / total:.1f}%)"


def _total_avg(total: int, avg: float | None) -> str:
    if avg is None:
        return f"{total} total"
    return f"{total} total, {avg:.1f} avg"


def _metric_values(questions: list[QuestionResult], key: str) -> list[float]:
    return [
        value
        for value in (
            _coerce_float(question.latency.get(key)) for question in questions
        )
        if value is not None
    ]


def _sum_or_none(values: list[float]) -> float | None:
    return round(sum(values), 6) if values else None


def _avg_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 6)
    rank = (len(sorted_values) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return round(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight,
        6,
    )


def _number_or_zero(value: Any) -> float:
    number = _coerce_float(value)
    return number if number is not None else 0.0


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coalesce_float(value: Any, fallback: float | None) -> float | None:
    coerced = _coerce_float(value)
    return coerced if coerced is not None else fallback


def _avg(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _as_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _structured_sequences(sequences: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize saved agent responses for the HTML conversation view.

    Historical result files store provider messages as ``repr`` strings.  Keep
    those strings available for inspection, but recover common dictionary
    messages so the report can render roles, content, tool calls, and metadata
    as separate fields.
    """

    structured = []
    for sequence in sequences:
        if not isinstance(sequence, dict):
            continue
        messages = [
            _structured_message(response, index)
            for index, response in enumerate(sequence.get("responses") or [], start=1)
            if isinstance(response, dict)
        ]
        structured.append(
            {
                "description": _as_text(sequence.get("description") or "Sequence"),
                "messages": messages,
            }
        )
    return structured


def _structured_message(response: dict[str, Any], index: int) -> dict[str, Any]:
    raw_response = response.get("raw_response")
    parsed_response = response.get("parsed_response")
    raw_message = _literal_message(raw_response)

    role = response.get("role")
    content = response.get("content")
    kind = response.get("kind")
    reasoning = response.get("reasoning")
    tool_name = response.get("tool_name")
    tool_args = response.get("tool_args")
    tool_calls = response.get("tool_calls")
    metadata = {}
    if isinstance(raw_message, dict):
        role = role or raw_message.get("role")
        message_type = raw_message.get("type")
        tool_name = tool_name or raw_message.get("tool_name") or raw_message.get("name")
        tool_calls = tool_calls or raw_message.get("tool_calls")
        reasoning = reasoning or raw_message.get("reasoning") or raw_message.get("thinking")
        content = content if content not in (None, "") else raw_message.get("content")
        if message_type == "function_call_output":
            role = role or "tool"
            content = raw_message.get("output", content)
        elif "toolResult" in raw_message:
            role = role or "tool"
            content = raw_message.get("toolResult")
        elif message_type in {"function_call", "custom_tool_call"}:
            role = role or "assistant"
        metadata = {
            key: value
            for key, value in raw_message.items()
            if key
            not in {
                "role",
                "content",
                "output",
                "tool_calls",
                "toolResult",
                "tool_name",
                "name",
                "reasoning",
                "thinking",
                "tool_args",
                "kind",
            }
            and value not in (None, "", [], {})
        }

    parsed_text = _as_optional_text(parsed_response)
    if role is None:
        role, inferred_content = _role_and_content_from_parsed(parsed_text, index)
        content = content if content not in (None, "") else inferred_content

    is_tool_response = role == "tool" or kind == "tool_result"
    is_tool_call = not is_tool_response and bool(
        kind == "tool_call" or tool_name or tool_calls
    )
    if content in (None, ""):
        # An assistant may legitimately return only a thinking field. Preserve
        # its empty content as an empty message box instead of promoting the
        # provider repr/raw response into the visible final answer. Tool calls
        # retain their existing parsed-response normalization.
        content = (
            ""
            if role == "assistant" and not is_tool_call
            else parsed_text if parsed_text not in (None, "") else raw_response
        )

    return {
        "role": _as_text(role or f"Message {index}"),
        "kind": kind,
        "content": content,
        "reasoning": reasoning,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_calls": tool_calls,
        "parsed_response": parsed_text,
        "metadata": None if is_tool_response else metadata or None,
        "raw_message": (
            raw_response
            if isinstance(role, str) and role.lower().startswith("assistant")
            else None
        ),
    }


def _literal_message(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return _model_repr_message(value)
    return parsed


_MODEL_REPR_PATTERN = re.compile(
    r"^role=(?P<role>.+?) content=(?P<content>.+?) thinking=(?P<thinking>.+?) "
    r"images=(?P<images>.+?) tool_name=(?P<tool_name>.+?) "
    r"tool_calls=(?P<tool_calls>.+)$",
    re.DOTALL,
)


def _model_repr_message(value: str) -> dict[str, Any] | None:
    """Recover fields from saved Pydantic-style model response reprs."""

    match = _MODEL_REPR_PATTERN.match(value)
    if match is None:
        return None

    message = {
        key: _literal_repr_value(match.group(key))
        for key in ("role", "content", "thinking", "images", "tool_name")
    }
    message["tool_calls"] = _tool_calls_from_repr(match.group("tool_calls"))
    return message


def _literal_repr_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _tool_calls_from_repr(value: str) -> Any:
    literal_value = _literal_repr_value(value)
    if not isinstance(literal_value, str):
        return literal_value

    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        return value

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Function":
            continue
        function = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                function[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                continue
        if function:
            calls.append({"function": function})
    return calls or value


def _role_and_content_from_parsed(
    parsed_response: str | None,
    index: int,
) -> tuple[str, str | None]:
    if not parsed_response:
        return f"Message {index}", parsed_response
    prefix, separator, remainder = parsed_response.partition(":")
    normalized = prefix.strip().lower()
    if separator and normalized in {
        "user",
        "assistant",
        "system",
        "developer",
        "tool",
    }:
        return normalized, remainder.lstrip()
    if normalized in {"function result", "tool result"}:
        return "tool", remainder.lstrip()
    if parsed_response.lstrip().lower().startswith("function call:"):
        return "assistant / tool call", parsed_response
    return "assistant", parsed_response


def _html_payload(sources: Sequence[ResultSource]) -> dict[str, Any]:
    source_payload = []
    overview = []
    provider_models = []
    questions = []

    for source_index, source in enumerate(sources):
        source_id = f"source-{source_index}"
        source_payload.append(
            {
                "id": source_id,
                "path": str(source.path),
                "metadata": source.metadata,
            }
        )
        for configuration in source.configurations:
            config_id = f"{source_id}:{configuration.configuration_name}"
            summary = asdict(configuration.summary)
            summary.update(_summarize_warmup(configuration.local_resources))
            provider_model_label = _provider_model_label(configuration.provider_models)
            providers = _provider_labels(configuration.provider_models)
            provider_label = ", ".join(providers)
            overview.append(
                {
                    "source_id": source_id,
                    "source_path": str(source.path),
                    "configuration": configuration.configuration_name,
                    "provider": provider_label,
                    "providers": providers,
                    "provider_model": provider_model_label,
                    "summary": summary,
                    "overlap_summaries": {
                        overlap: {
                            "summary": asdict(summarize_configuration(subset)),
                            "local_resources_summary": _summarize_local_resources(None, subset),
                        }
                        for overlap in ("direct", "related", "absent", "unclassified")
                        if (subset := [
                            question for question in configuration.questions
                            if (question.overlap_class or "unclassified") == overlap
                        ])
                    },
                    "analysis_summary": configuration.analysis_summary,
                    "cost_summary": configuration.cost_summary,
                    "local_resources": configuration.local_resources,
                    "local_resources_summary": _summarize_local_resources(
                        configuration.local_resources,
                        configuration.questions,
                    ),
                }
            )
            for ref in configuration.provider_models:
                request_parameters = ref.request_parameters or {}
                parameter_capabilities = ref.parameter_capabilities or {}
                reasoning = _as_dict_or_none(
                    request_parameters.get("reasoning")
                ) or {}
                provider_models.append(
                    {
                        "source_id": source_id,
                        "source_path": str(source.path),
                        "configuration": configuration.configuration_name,
                        "phase": ref.phase,
                        "provider": ref.provider,
                        "model_identifier": ref.model_identifier,
                        "temperature": request_parameters.get("temperature"),
                        "seed": request_parameters.get("seed"),
                        "reasoning_mode": reasoning.get("mode"),
                        "reasoning_effort": reasoning.get("effort"),
                        "reasoning_support": parameter_capabilities.get("reasoning"),
                        "reasoning_supported_efforts": ", ".join(
                            parameter_capabilities.get("reasoning_efforts") or []
                        ),
                        "temperature_supported": parameter_capabilities.get(
                            "temperature"
                        ),
                        "seed_supported": parameter_capabilities.get("seed"),
                        "require_parameters": ref.require_parameters,
                        "provider_model": "/".join(
                            part
                            for part in [ref.provider, ref.model_identifier]
                            if part
                        ),
                        "configuration_provider_model": provider_model_label,
                    }
                )
            for question_index, question in enumerate(configuration.questions):
                questions.append(
                    {
                        "id": f"{config_id}:question-{question_index}",
                        "source_id": source_id,
                        "source_path": str(source.path),
                        "configuration": configuration.configuration_name,
                        "provider": provider_label,
                        "providers": providers,
                        "provider_model": question.provider_model,
                        "name": question.name,
                        "question_type": question.question_type,
                        "overlap_class": question.overlap_class,
                        "tags": question.tags,
                        "question": question.question,
                        "solution": question.solution,
                        "answer": question.answer,
                        "completed": question.completed,
                        "valid_answer_format": question.valid_answer_format,
                        "correct": question.correct,
                        "final_answer_match": question.final_answer_match,
                        "cypher_solution_match": question.cypher_solution_match,
                        "tool_executable": question.tool_executable,
                        "generated_cypher": question.generated_cypher,
                        "cypher_tool_output": question.cypher_tool_output,
                        "cypher_validation_issues": question.cypher_validation_issues,
                        "input_tokens": question.input_tokens,
                        "cached_input_tokens": question.cached_input_tokens,
                        "output_tokens": question.output_tokens,
                        "reasoning_tokens": question.reasoning_tokens,
                        "n_tool_calls": question.n_tool_calls,
                        "latency": question.latency,
                        "cost": question.cost,
                        "local_resources": question.local_resources,
                        "n_sequences": question.n_sequences,
                        "sequences": _structured_sequences(question.sequences),
                    }
                )

    return {
        "sources": source_payload,
        "overview": overview,
        "provider_models": provider_models,
        "questions": questions,
        "artifacts": _artifact_payload(sources),
    }


def _artifact_payload(sources: Sequence[ResultSource]) -> list[dict[str, Any]]:
    artifacts = []
    seen_paths: set[Path] = set()

    def add_artifact(
        artifact: str,
        value: str | Path,
        *,
        base_dir: Path | None = None,
        scan_references: bool = False,
    ) -> None:
        path = _resolve_artifact_path(value, base_dir=base_dir)
        if path in seen_paths:
            return
        seen_paths.add(path)
        artifacts.append({"artifact": artifact, **_artifact_file_info(path)})

        if (
            not scan_references
            or not path.is_file()
            or path.suffix.lower()
            not in {
                ".yaml",
                ".yml",
            }
        ):
            return
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return
        for reference_name, reference_value in _iter_artifact_references(document):
            add_artifact(
                reference_name,
                reference_value,
                base_dir=path.parent,
                scan_references=True,
            )

    for source in sources:
        add_artifact("result", source.path)
        benchmark_manifest = source.metadata.get("benchmark_manifest")
        if benchmark_manifest:
            add_artifact(
                "benchmark_manifest",
                str(benchmark_manifest),
                scan_references=True,
            )
            # Runtime sweep YAMLs are temporary implementation details. The
            # repository-owned benchmark manifest is the reproducible source.
            continue
        source_experiment = source.metadata.get("source_experiment")
        if source_experiment:
            add_artifact("source_experiment", str(source_experiment), scan_references=True)
    return artifacts


def _iter_artifact_references(value: Any, key: str = "file"):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _iter_artifact_references(child_value, str(child_key))
    elif isinstance(value, list):
        for child_value in value:
            yield from _iter_artifact_references(child_value, key)
    elif isinstance(value, str):
        suffix = Path(value.replace("${HERACLES_AGENTS_PATH}", "")).suffix.lower()
        if suffix in _ARTIFACT_FILE_SUFFIXES:
            yield key, value


def _resolve_artifact_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    expanded = str(value).replace("${HERACLES_AGENTS_PATH}", str(_PROJECT_ROOT))
    expanded = expanded.replace("$HERACLES_AGENTS_PATH", str(_PROJECT_ROOT))
    benchmark_root_value = os.environ.get("HERACLES_BENCHMARK_PATH")
    if benchmark_root_value:
        expanded = expanded.replace(
            "${HERACLES_BENCHMARK_PATH}", benchmark_root_value
        ).replace("$HERACLES_BENCHMARK_PATH", benchmark_root_value)
    candidate = Path(os.path.expandvars(expanded)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / candidate)
    if benchmark_root_value:
        candidates.append(Path(benchmark_root_value) / candidate)
    candidates.extend((Path.cwd() / candidate, _PROJECT_ROOT / candidate))
    return next(
        (path.resolve() for path in candidates if path.exists()),
        candidates[0].resolve(),
    )


def _artifact_file_info(path: Path) -> dict[str, Any]:
    display_path = str(path)
    display_roots = [_PROJECT_ROOT]
    if os.environ.get("HERACLES_BENCHMARK_PATH"):
        display_roots.insert(0, Path(os.environ["HERACLES_BENCHMARK_PATH"]))
    for root in display_roots:
        try:
            display_path = str(path.relative_to(root))
            break
        except ValueError:
            continue
    info: dict[str, Any] = {"path": display_path, "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        info.update(
            {
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    timezone.utc,
                ).isoformat(),
            }
        )
    return info


def _summarize_local_resources(
    configuration_local_resources: dict[str, Any] | None,
    questions: Sequence[QuestionResult],
) -> dict[str, Any]:
    if configuration_local_resources:
        return {
            "cpu_avg_percent": _first_non_none(
                _local_resource_value(
                    configuration_local_resources,
                    "cpu",
                    "container_cpu_percent_normalized_avg",
                ),
                _local_resource_value(
                    configuration_local_resources,
                    "cpu",
                    "container_cpu_percent_avg",
                ),
            ),
            "cpu_peak_percent": _first_non_none(
                _local_resource_value(
                    configuration_local_resources,
                    "cpu",
                    "container_cpu_percent_normalized_peak",
                ),
                _local_resource_value(
                    configuration_local_resources,
                    "cpu",
                    "container_cpu_percent_peak",
                ),
            ),
            "ram_peak_bytes": _local_resource_value(
                configuration_local_resources,
                "ram",
                "container_ram_cgroup_working_set_bytes_peak",
            ),
            "gpu_vram_peak_mib": _local_resource_value(
                configuration_local_resources,
                "gpu",
                "gpu_memory_used_mib_adjusted_peak",
            ),
            "gpu_avg_percent": _local_resource_value(
                configuration_local_resources,
                "gpu",
                "gpu_utilization_percent_raw_avg",
            ),
            "gpu_power_avg_w": _local_resource_value(
                configuration_local_resources,
                "gpu",
                "gpu_power_w_adjusted_avg",
            ),
            "gpu_energy_wh": _local_resource_value(
                configuration_local_resources,
                "gpu",
                "gpu_energy_wh_adjusted",
            ),
        }

    def values(group: str, key: str) -> list[float]:
        return [
            value
            for question in questions
            for value in [
                _coerce_float(
                    ((question.local_resources or {}).get(group) or {}).get(key)
                )
            ]
            if value is not None
        ]

    cpu_avg = values("cpu", "container_cpu_percent_avg")
    cpu_peak = values("cpu", "container_cpu_percent_peak")
    cpu_normalized_avg = values("cpu", "container_cpu_percent_normalized_avg")
    cpu_normalized_peak = values("cpu", "container_cpu_percent_normalized_peak")
    ram_peak = values("ram", "container_ram_cgroup_working_set_bytes_peak")
    gpu_vram_peak = values("gpu", "gpu_memory_used_mib_adjusted_peak")
    gpu_util_avg = values("gpu", "gpu_utilization_percent_raw_avg")
    gpu_power_avg = values("gpu", "gpu_power_w_adjusted_avg")
    gpu_energy = values("gpu", "gpu_energy_wh_adjusted")

    cpu_avg_value = _avg(cpu_normalized_avg)
    if cpu_avg_value is None:
        cpu_avg_value = _avg(cpu_avg)

    return {
        "cpu_avg_percent": cpu_avg_value,
        "cpu_peak_percent": (
            max(cpu_normalized_peak)
            if cpu_normalized_peak
            else max(cpu_peak)
            if cpu_peak
            else None
        ),
        "ram_peak_bytes": max(ram_peak) if ram_peak else None,
        "gpu_vram_peak_mib": max(gpu_vram_peak) if gpu_vram_peak else None,
        "gpu_avg_percent": _avg(gpu_util_avg),
        "gpu_power_avg_w": _avg(gpu_power_avg),
        "gpu_energy_wh": round(sum(gpu_energy), 9) if gpu_energy else None,
    }


def _summarize_warmup(
    configuration_local_resources: dict[str, Any] | None,
) -> dict[str, Any]:
    warmup = (
        (configuration_local_resources or {}).get("warmup") or {}
    )
    requests = warmup.get("requests") or []
    cold_start = next(
        (
            request
            for request in requests
            if request.get("kind") == "cold_start" and request.get("succeeded")
        ),
        None,
    )
    return {
        "cold_start_load_duration_seconds": _coerce_float(
            (cold_start or {}).get("load_duration_seconds")
        ),
        "cold_start_total_duration_seconds": _coerce_float(
            (cold_start or {}).get("total_duration_seconds")
        ),
        "warmup_resident_verified": warmup.get("resident_verified"),
    }


def _local_resource_value(
    local_resources: dict[str, Any],
    group: str,
    key: str,
) -> float | None:
    return _coerce_float((local_resources.get(group) or {}).get(key))


def _first_non_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heracles Experiment Results</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #1d2430;
      --muted: #667085;
      --accent: #0f766e;
      --accent-soft: #e1f3f1;
      --good: #117a37;
      --good-bg: #e7f6ec;
      --bad: #b42318;
      --bad-bg: #fde7e4;
      --warn: #a15c00;
      --warn-bg: #fff3d6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .page-header {
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    header { padding: 18px 24px 10px; }
    h1 { font-size: 22px; margin: 0 0 4px; }
    h2 { font-size: 20px; margin: 0 0 12px; }
    h3 { font-size: 15px; margin: 0 0 12px; }
    .muted { color: var(--muted); }
    .wrap { padding: 22px 24px 28px; }
    .report-nav {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      padding: 0 24px;
    }
    .tabs {
      display: flex;
      gap: 6px;
      overflow-x: auto;
    }
    .tab-button {
      appearance: none;
      border: 1px solid transparent;
      border-bottom: 0;
      background: transparent;
      color: var(--muted);
      padding: 10px 12px;
      font: inherit;
      cursor: pointer;
    }
    .tab-button.active {
      color: var(--accent);
      background: var(--accent-soft);
      border-color: var(--line);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .metric-controls {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 12px;
      padding: 0 0 9px;
      margin-left: auto;
    }
    .metric-controls[hidden] { display: none; }
    .metric-controls label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      color: var(--text);
      font-size: 13px;
    }
    .metric-controls input { margin: 0; }
    .table-wrap {
      max-width: 100%;
      overflow-x: auto;
    }
    .section-body { margin-top: 10px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #eef1f5;
      cursor: pointer;
      white-space: nowrap;
      user-select: none;
    }
    th.sort-asc::after { content: " ▲"; color: var(--accent); }
    th.sort-desc::after { content: " ▼"; color: var(--accent); }
    .filter-row th {
      background: var(--panel);
      cursor: default;
      padding: 6px 8px;
    }
    .column-filter {
      width: 100%;
      min-width: 96px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      padding: 5px 7px;
      font: inherit;
      font-size: 12px;
    }
    td {
      max-width: 340px;
      overflow-wrap: anywhere;
    }
    tr[data-question-id] { cursor: pointer; }
    tr[data-question-id]:hover { background: #f2f6f5; }
    tr[data-question-id].active { background: var(--accent-soft); }
    .status-pill {
      display: inline-block;
      border-radius: 999px;
      padding: 1px 8px 2px;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.5;
      white-space: nowrap;
    }
    .status-good { color: var(--good); background: var(--good-bg); }
    .status-bad { color: var(--bad); background: var(--bad-bg); }
    .status-warn { color: var(--warn); background: var(--warn-bg); }
    .status-null { color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 10px 12px 12px;
    }
    .detail {
      position: sticky;
      top: 118px;
      max-height: calc(100vh - 140px);
      overflow: auto;
    }
    .detail h4 { margin: 16px 0 8px; font-size: 13px; }
    .detail h2:first-child { margin-top: 0; }
    .sequence-group { margin-top: 18px; }
    .sequence-group h4 { margin: 0 0 8px; }
    .message-list { display: grid; gap: 10px; }
    details {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 10px 12px;
    }
    summary { cursor: pointer; font-weight: 650; }
    .validation-detail-group { margin: 10px 0; }
    .validation-detail-group > summary {
      color: var(--accent);
      font-size: 14px;
    }
    .validation-detail-group-body {
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 2px;
    }
    .message-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px 12px;
    }
    .message-role {
      font-weight: 700;
      margin-bottom: 8px;
      color: #394656;
    }
    .message-card pre { margin: 0; }
    .message-card pre + h5 { margin-top: 12px; }
    .message-card h5 { margin: 10px 0 6px; color: var(--muted); }
    .message-card details {
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 8px;
    }
    .message-card summary { cursor: pointer; font-weight: 650; color: var(--muted); }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f2f4f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      max-height: 260px;
      overflow: auto;
      margin: 0;
      font: inherit;
    }
    .kv {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 6px 10px;
      margin: 8px 0;
    }
    .kv dt { color: var(--muted); font-weight: 650; }
    .kv dd { margin: 0; overflow-wrap: anywhere; }
    .empty {
      padding: 16px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
    }
    @media (max-width: 980px) {
      .report-nav { align-items: stretch; flex-direction: column; gap: 4px; }
      .metric-controls { justify-content: flex-start; margin-left: 0; }
      .grid { grid-template-columns: 1fr; }
      .detail { position: static; max-height: none; }
      th { position: static; }
    }
  </style>
</head>
<body>
  <div class="page-header">
    <header>
      <h1>Heracles Experiment Results</h1>
      <div class="muted">Generated __GENERATED_AT__</div>
    </header>
    <div class="report-nav">
      <nav class="tabs" role="tablist" aria-label="Experiment report tabs">
        <button class="tab-button active" type="button" role="tab" aria-selected="true" aria-controls="providers-tab" data-tab="providers-tab">Provider Models</button>
        <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="overview-tab" data-tab="overview-tab">Overview</button>
        <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="questions-tab" data-tab="questions-tab">Questions</button>
        <button class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="artifacts-tab" data-tab="artifacts-tab">Artifacts</button>
      </nav>
      <div class="metric-controls" aria-label="Metric columns" hidden>
        <label for="overlap-class-filter">Task overlap
          <select id="overlap-class-filter">
            <option value="">All classes</option>
            <option value="direct">Direct overlap</option>
            <option value="related">Related variant / composition</option>
            <option value="absent">Absent from training catalogs</option>
            <option value="unclassified">Unclassified</option>
          </select>
        </label>
        <label><input type="radio" name="metric-group" data-group="quality" checked> Quality</label>
        <label><input type="radio" name="metric-group" data-group="tokens"> Tokens/tools</label>
        <label><input type="radio" name="metric-group" data-group="latency"> Latency</label>
        <label><input type="radio" name="metric-group" data-group="cost"> Cost</label>
        <label><input type="radio" name="metric-group" data-group="local_resources"> Local Resources</label>
      </div>
    </div>
  </div>
  <main class="wrap">
    <section id="providers-tab" class="tab-panel active" role="tabpanel">
      <h2>Provider Models</h2>
      <div class="section-body" id="providerModels"></div>
    </section>
    <section id="overview-tab" class="tab-panel" role="tabpanel">
      <h2>Overview</h2>
      <div class="section-body" id="overview"></div>
    </section>
    <section id="questions-tab" class="tab-panel" role="tabpanel">
      <h2>Questions</h2>
      <div class="section-body grid">
        <div id="questions"></div>
        <aside id="detail" class="panel detail">
          <div class="muted">Select a question row to inspect details.</div>
        </aside>
      </div>
    </section>
    <section id="artifacts-tab" class="tab-panel" role="tabpanel">
      <h2>Artifacts</h2>
      <div class="section-body" id="artifacts"></div>
    </section>
  </main>
  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const state = { selectedId: null };
    const groupInputs = [...document.querySelectorAll("[data-group]")];
    const tabButtons = [...document.querySelectorAll(".tab-button")];
    const tabPanels = [...document.querySelectorAll(".tab-panel")];
    const metricControls = document.querySelector(".metric-controls");
    const overlapFilter = document.getElementById("overlap-class-filter");

    function overlapLabel(value) {
      return {
        direct: "Direct overlap",
        related: "Related variant / composition",
        absent: "Absent from training catalogs",
        unclassified: "Unclassified",
      }[value] || "All classes";
    }
    function overviewForOverlap() {
      if (!overlapFilter.value) return report.overview;
      return report.overview.flatMap(row => {
        const subset = (row.overlap_summaries || {})[overlapFilter.value];
        return subset ? [{ ...row, ...subset, overlap_class: overlapFilter.value }] : [];
      });
    }
    function questionsForOverlap() {
      return report.questions.filter(question => !overlapFilter.value ||
        (question.overlap_class || "unclassified") === overlapFilter.value);
    }

    function text(value) {
      if (value === null || value === undefined) return "";
      return String(value);
    }
    function number(value, digits = 3) {
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (!Number.isFinite(n)) return text(value);
      return n.toFixed(digits).replace(/0+$/, "").replace(/\\.$/, "");
    }
    function seconds(value) {
      const formatted = number(value);
      return formatted === "" ? "" : `${formatted}s`;
    }
    function percent(value) {
      const formatted = number(value, 1);
      return formatted === "" ? "" : `${formatted}%`;
    }
    function money(value) {
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (!Number.isFinite(n)) return text(value);
      return "$" + n.toFixed(6).replace(/0+$/, "").replace(/\\.$/, "");
    }
    function bytes(value) {
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (!Number.isFinite(n)) return text(value);
      const mibValue = n / (1024 * 1024);
      return `${number(mibValue, 1)} MiB`;
    }
    function mib(value) {
      const formatted = number(value, 1);
      return formatted === "" ? "" : `${formatted} MiB`;
    }
    function watts(value) {
      const formatted = number(value, 2);
      return formatted === "" ? "" : `${formatted} W`;
    }
    function wattHours(value) {
      const formatted = number(value, 4);
      return formatted === "" ? "" : `${formatted} Wh`;
    }
    function tokensPerSecond(value) {
      const formatted = number(value, 2);
      return formatted === "" ? "" : `${formatted} tok/s`;
    }
    function bool(value) {
      if (value === true) return '<span class="status-pill status-good">True</span>';
      if (value === false) return '<span class="status-pill status-bad">False</span>';
      return '<span class="status-null"></span>';
    }
    function escapeHtml(value) {
      return text(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function metric(q, key) {
      return q.latency && q.latency[key] !== undefined ? q.latency[key] : null;
    }
    function cost(q) {
      return q.cost && q.cost.total_cost_usd !== undefined ? q.cost.total_cost_usd : null;
    }
    function questionThroughput(q) {
      const calls = q.cost && Array.isArray(q.cost.llm_calls) ? q.cost.llm_calls : [];
      const timed = calls.filter(call => Number(
        call.generation_time_seconds || call.observed_call_seconds
      ) > 0);
      if (timed.length) {
        const seconds = timed.reduce((total, call) => total + Number(
          call.generation_time_seconds || call.observed_call_seconds
        ), 0);
        const tokens = timed.reduce((total, call) => total + Number(call.output_tokens || 0), 0);
        return seconds > 0 ? tokens / seconds : null;
      }
      return metric(q, "output_tokens_per_second");
    }
    function localSummary(row, key) {
      return row.local_resources_summary ? row.local_resources_summary[key] : null;
    }
    function firstRecordedValue(...values) {
      return values.find(value => value !== null && value !== undefined && value !== "");
    }
    function questionLocalResource(row, group, ...keys) {
      const resources = row.local_resources || {};
      const values = resources[group] || {};
      return firstRecordedValue(...keys.map(key => values[key]));
    }
    function questionLocalSummary(row, key) {
      const accessors = {
        cpu_avg_percent: () => questionLocalResource(row, "cpu", "container_cpu_percent_normalized_avg", "container_cpu_percent_avg"),
        cpu_peak_percent: () => questionLocalResource(row, "cpu", "container_cpu_percent_normalized_peak", "container_cpu_percent_peak"),
        ram_peak_bytes: () => questionLocalResource(row, "ram", "container_ram_cgroup_working_set_bytes_peak"),
        gpu_vram_peak_mib: () => questionLocalResource(row, "gpu", "gpu_memory_used_mib_adjusted_peak"),
        gpu_avg_percent: () => questionLocalResource(row, "gpu", "gpu_utilization_percent_raw_avg"),
        gpu_power_avg_w: () => questionLocalResource(row, "gpu", "gpu_power_w_adjusted_avg"),
        gpu_energy_wh: () => questionLocalResource(row, "gpu", "gpu_energy_wh_adjusted"),
      };
      return accessors[key] ? accessors[key]() : null;
    }
    function hasRecordedValue(value) {
      return value !== null && value !== undefined && value !== "";
    }
    function hasMeaningfulValue(value) {
      if (value === null || value === undefined || value === "") return false;
      if (typeof value === "number") return value !== 0;
      return true;
    }
    function hasValue(accessor) {
      return row => hasMeaningfulValue(accessor(row));
    }
    function hasRecorded(accessor) {
      return row => hasRecordedValue(accessor(row));
    }
    function activeMetricGroup() {
      const active = groupInputs.find(input => input.checked);
      return active ? active.dataset.group : null;
    }
    function columnMatchesActiveGroup(column) {
      return !column.group || column.group === activeMetricGroup();
    }
    function rowsForActiveMetricGroup(rows, tableKind) {
      const group = activeMetricGroup();
      if (group === "cost") {
        return rows.filter(row => hasRecordedValue(
          tableKind === "overview" ? row.summary.cost_total_usd : cost(row)
        ));
      }
      if (group === "local_resources") {
        return rows.filter(row => {
          const summary = tableKind === "overview"
            ? row.local_resources_summary || {}
            : {
                cpu: questionLocalSummary(row, "cpu_avg_percent"),
                ram: questionLocalSummary(row, "ram_peak_bytes"),
                gpu: questionLocalSummary(row, "gpu_avg_percent"),
                power: questionLocalSummary(row, "gpu_power_avg_w"),
                energy: questionLocalSummary(row, "gpu_energy_wh"),
              };
          return Object.values(summary).some(hasRecordedValue);
        });
      }
      return rows;
    }
    function renderTable(target, columns, rows, rowAttrs = () => "") {
      const visibleColumns = columns.filter(column =>
        columnMatchesActiveGroup(column) &&
        (!column.visible || rows.some(row => column.visible(row)))
      );
      if (!rows.length) {
        target.innerHTML = '<div class="empty">No rows available.</div>';
        return;
      }
      const header = visibleColumns.map((column, index) =>
        `<th data-sort-index="${index}" aria-sort="none">${escapeHtml(column.label)}</th>`
      ).join("");
      const filterRow = visibleColumns.map((column, index) => `
        <th>
          <input
            class="column-filter"
            type="search"
            data-filter-index="${index}"
            placeholder="Filter ${escapeHtml(column.label)}"
            aria-label="Filter ${escapeHtml(column.label)}"
          >
        </th>
      `).join("");
      const body = rows.map(row => {
        const cells = visibleColumns.map(column => {
          const sortValue = column.sortValue
            ? column.sortValue(row)
            : valueByPath(row, column.key);
          const rendered = column.render
            ? column.render(row)
            : escapeHtml(valueByPath(row, column.key));
          return `<td data-sort-value="${escapeHtml(sortValue)}">${rendered}</td>`;
        }).join("");
        return `<tr ${rowAttrs(row)}>${cells}</tr>`;
      }).join("");
      target.innerHTML = `
        <div class="table-wrap">
          <table class="data-table sortable filterable">
            <thead><tr>${header}</tr><tr class="filter-row">${filterRow}</tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      bindTableInteractions(target.querySelector("table"));
    }
    function bindTableInteractions(table) {
      if (!table) return;
      table.querySelectorAll("th[data-sort-index]").forEach(th => {
        th.addEventListener("click", () => {
          const index = Number(th.dataset.sortIndex);
          const direction = th.classList.contains("sort-asc") ? -1 : 1;
          table.querySelectorAll("th[data-sort-index]").forEach(header => {
            header.classList.remove("sort-asc", "sort-desc");
            header.setAttribute("aria-sort", "none");
          });
          th.classList.add(direction === 1 ? "sort-asc" : "sort-desc");
          th.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
          const body = table.tBodies[0];
          [...body.rows]
            .sort((left, right) => compareSortValues(
              left.cells[index] && left.cells[index].dataset.sortValue,
              right.cells[index] && right.cells[index].dataset.sortValue,
              direction
            ))
            .forEach(row => body.appendChild(row));
        });
      });
      table.querySelectorAll(".column-filter").forEach(input => {
        input.addEventListener("input", () => applyColumnFilters(table));
      });
    }
    function compareSortValues(av, bv, direction) {
      const aMissing = av === null || av === undefined || av === "";
      const bMissing = bv === null || bv === undefined || bv === "";
      if (aMissing && bMissing) return 0;
      if (aMissing) return 1;
      if (bMissing) return -1;
      const an = Number(av);
      const bn = Number(bv);
      if (Number.isFinite(an) && Number.isFinite(bn)) {
        return (an - bn) * direction;
      }
      return String(av).localeCompare(String(bv), undefined, { numeric: true }) * direction;
    }
    function applyColumnFilters(table) {
      const terms = [...table.querySelectorAll(".column-filter")].map(input => ({
        index: Number(input.dataset.filterIndex),
        value: input.value.trim().toLowerCase(),
      }));
      [...table.tBodies[0].rows].forEach(row => {
        const visible = terms.every(term => {
          if (!term.value) return true;
          const cell = row.cells[term.index];
          if (!cell) return false;
          const haystack = `${cell.dataset.sortValue || ""} ${cell.textContent || ""}`.toLowerCase();
          return haystack.includes(term.value);
        });
        row.hidden = !visible;
      });
      const questionRows = [...table.querySelectorAll("tr[data-question-id]")];
      if (questionRows.length) {
        const selected = questionRows.find(row => row.classList.contains("active"));
        if (!selected || selected.hidden) {
          selectQuestionRow(questionRows.find(row => !row.hidden) || null);
        }
      }
    }
    function valueByPath(row, path) {
      return path.split(".").reduce((value, key) => {
        if (value === null || value === undefined) return undefined;
        return value[key];
      }, row);
    }
    function renderOverview() {
      const columns = [
        { key: "source_path", label: "Source" },
        { key: "configuration", label: "Configuration" },
        { key: "provider", label: "Provider" },
        { key: "provider_model", label: "Provider / Model" },
        { key: "overlap_class", label: "Task Overlap", render: r => escapeHtml(overlapLabel(r.overlap_class)) },
        { key: "questions", label: "Questions", group: "quality", render: r => text(r.summary.questions), sortValue: r => r.summary.questions },
        { key: "tool_executable_rate", label: "Tool Executable", group: "quality", render: r => percent((r.summary.tool_executable_rate || 0) * 100), sortValue: r => r.summary.tool_executable_rate, visible: hasRecorded(r => r.summary.tool_executable_rate) },
        { key: "cypher_solution_match_rate", label: "Cypher Solution / Grounding Match", group: "quality", render: r => percent((r.summary.cypher_solution_match_rate || 0) * 100), sortValue: r => r.summary.cypher_solution_match_rate, visible: hasRecorded(r => r.summary.cypher_solution_match_rate) },
        { key: "final_answer_match_rate", label: "Final Answer Match", group: "quality", render: r => percent((r.summary.final_answer_match_rate || 0) * 100), sortValue: r => r.summary.final_answer_match_rate },
        { key: "input_tokens_total", label: "Input Tokens", group: "tokens", render: r => text(r.summary.input_tokens_total), sortValue: r => r.summary.input_tokens_total, visible: hasValue(r => r.summary.input_tokens_total) },
        { key: "cached_input_tokens_total", label: "Cached Input Tokens", group: "tokens", render: r => text(r.summary.cached_input_tokens_total), sortValue: r => r.summary.cached_input_tokens_total },
        { key: "output_tokens_total", label: "Output Tokens", group: "tokens", render: r => text(r.summary.output_tokens_total), sortValue: r => r.summary.output_tokens_total, visible: hasValue(r => r.summary.output_tokens_total) },
        { key: "reasoning_tokens_total", label: "Reasoning Tokens", group: "tokens", render: r => text(r.summary.reasoning_tokens_total), sortValue: r => r.summary.reasoning_tokens_total },
        { key: "tool_calls_total", label: "Tool Calls", group: "tokens", render: r => text(r.summary.tool_calls_total), sortValue: r => r.summary.tool_calls_total, visible: hasValue(r => r.summary.tool_calls_total) },
        { key: "output_tokens_per_second", label: "Throughput", group: "latency", render: r => tokensPerSecond(r.summary.output_tokens_per_second), sortValue: r => r.summary.output_tokens_per_second, visible: hasRecorded(r => r.summary.output_tokens_per_second) },
        { key: "end_to_end_latency_avg", label: "Average End-to-End Latency", group: "latency", render: r => seconds(r.summary.end_to_end_latency_avg), sortValue: r => r.summary.end_to_end_latency_avg, visible: hasValue(r => r.summary.end_to_end_latency_avg) },
        { key: "cold_start_load_duration_seconds", label: "Cold-start Load Duration", group: "latency", render: r => seconds(r.summary.cold_start_load_duration_seconds), sortValue: r => r.summary.cold_start_load_duration_seconds, visible: hasValue(r => r.summary.cold_start_load_duration_seconds) },
        { key: "load_duration_total", label: "Measured Load Duration", group: "latency", render: r => seconds(r.summary.load_duration_total), sortValue: r => r.summary.load_duration_total, visible: hasValue(r => r.summary.load_duration_total) },
        { key: "warmup_resident_verified", label: "Warmup Resident", group: "latency", render: r => bool(r.summary.warmup_resident_verified), sortValue: r => r.summary.warmup_resident_verified, visible: hasRecorded(r => r.summary.warmup_resident_verified) },
        { key: "cost_total_usd", label: "Cost", group: "cost", render: r => money(r.summary.cost_total_usd), sortValue: r => r.summary.cost_total_usd, visible: hasValue(r => r.summary.cost_total_usd) },
        { key: "cost_per_correct_answer_usd", label: "Cost per Success", group: "cost", render: r => money(r.summary.cost_per_correct_answer_usd), sortValue: r => r.summary.cost_per_correct_answer_usd, visible: hasRecorded(r => r.summary.cost_per_correct_answer_usd) },
        { key: "local_resources_summary.cpu_avg_percent", label: "CPU Avg", group: "local_resources", render: r => percent(localSummary(r, "cpu_avg_percent")), sortValue: r => localSummary(r, "cpu_avg_percent"), visible: hasValue(r => localSummary(r, "cpu_avg_percent")) },
        { key: "local_resources_summary.cpu_peak_percent", label: "CPU Peak", group: "local_resources", render: r => percent(localSummary(r, "cpu_peak_percent")), sortValue: r => localSummary(r, "cpu_peak_percent"), visible: hasValue(r => localSummary(r, "cpu_peak_percent")) },
        { key: "local_resources_summary.ram_peak_bytes", label: "RAM Peak", group: "local_resources", render: r => bytes(localSummary(r, "ram_peak_bytes")), sortValue: r => localSummary(r, "ram_peak_bytes"), visible: hasValue(r => localSummary(r, "ram_peak_bytes")) },
        { key: "local_resources_summary.gpu_vram_peak_mib", label: "VRAM Peak", group: "local_resources", render: r => mib(localSummary(r, "gpu_vram_peak_mib")), sortValue: r => localSummary(r, "gpu_vram_peak_mib"), visible: hasValue(r => localSummary(r, "gpu_vram_peak_mib")) },
        { key: "local_resources_summary.gpu_avg_percent", label: "GPU Avg", group: "local_resources", render: r => percent(localSummary(r, "gpu_avg_percent")), sortValue: r => localSummary(r, "gpu_avg_percent"), visible: hasValue(r => localSummary(r, "gpu_avg_percent")) },
        { key: "local_resources_summary.gpu_power_avg_w", label: "GPU Power Avg", group: "local_resources", render: r => watts(localSummary(r, "gpu_power_avg_w")), sortValue: r => localSummary(r, "gpu_power_avg_w"), visible: hasValue(r => localSummary(r, "gpu_power_avg_w")) },
        { key: "local_resources_summary.gpu_energy_wh", label: "GPU Energy", group: "local_resources", render: r => wattHours(localSummary(r, "gpu_energy_wh")), sortValue: r => localSummary(r, "gpu_energy_wh"), visible: hasValue(r => localSummary(r, "gpu_energy_wh")) },
      ];
      renderTable(
        document.getElementById("overview"),
        columns,
        rowsForActiveMetricGroup(overviewForOverlap(), "overview")
      );
    }
    function renderProviderModels() {
      const columns = [
        { key: "source_path", label: "Source" },
        { key: "configuration", label: "Configuration" },
        { key: "phase", label: "Phase" },
        { key: "provider", label: "Provider" },
        { key: "model_identifier", label: "Model Identifier" },
        { key: "reasoning_support", label: "Reasoning Support" },
        { key: "reasoning_mode", label: "Reasoning Mode" },
        { key: "reasoning_effort", label: "Reasoning Effort" },
      ];
      renderTable(document.getElementById("providerModels"), columns, report.provider_models);
    }
    function renderArtifacts() {
      const columns = [
        { key: "artifact", label: "Artifact" },
        { key: "path", label: "Path" },
        { key: "exists", label: "Exists", render: r => bool(r.exists) },
        { key: "bytes", label: "Bytes", render: r => text(r.bytes), sortValue: r => r.bytes },
        { key: "modified_at", label: "Modified At" },
      ];
      renderTable(document.getElementById("artifacts"), columns, report.artifacts || []);
    }
    function renderQuestions() {
      const columns = [
        { key: "configuration", label: "Configuration" },
        { key: "provider", label: "Provider" },
        { key: "provider_model", label: "Provider / Model" },
        { key: "name", label: "Topic" },
        { key: "question_type", label: "Question Type", visible: hasRecorded(q => q.question_type) },
        { key: "overlap_class", label: "Task Overlap", render: q => escapeHtml(overlapLabel(q.overlap_class || "unclassified")) },
        { key: "tags", label: "Tags", render: q => escapeHtml((q.tags || []).join(", ")), visible: q => (q.tags || []).length > 0 },
        { key: "tool_executable", label: "Tool Executable", group: "quality", render: q => bool(q.tool_executable), visible: hasRecorded(q => q.tool_executable) },
        { key: "cypher_solution_match", label: "Cypher Solution / Grounding Match", group: "quality", render: q => bool(q.cypher_solution_match), visible: hasRecorded(q => q.cypher_solution_match) },
        { key: "final_answer_match", label: "Final Answer Match", group: "quality", render: q => bool(q.final_answer_match) },
        { key: "input_tokens", label: "Input Tokens", group: "tokens", visible: hasValue(q => q.input_tokens) },
        { key: "cached_input_tokens", label: "Cached Input Tokens", group: "tokens" },
        { key: "output_tokens", label: "Output Tokens", group: "tokens", visible: hasValue(q => q.output_tokens) },
        { key: "reasoning_tokens", label: "Reasoning Tokens", group: "tokens" },
        { key: "n_tool_calls", label: "Tool Calls", group: "tokens", visible: hasValue(q => q.n_tool_calls) },
        { key: "latency.output_tokens_per_second", label: "Throughput", group: "latency", render: q => tokensPerSecond(questionThroughput(q)), sortValue: q => questionThroughput(q), visible: hasRecorded(q => questionThroughput(q)) },
        { key: "latency.end_to_end_seconds", label: "End-to-End Latency", group: "latency", render: q => seconds(metric(q, "end_to_end_seconds")), visible: hasValue(q => metric(q, "end_to_end_seconds")) },
        { key: "latency.llm_call_seconds", label: "LLM Latency", group: "latency", render: q => seconds(metric(q, "llm_call_seconds")), visible: hasValue(q => metric(q, "llm_call_seconds")) },
        { key: "latency.tool_execution_seconds", label: "Tool Latency", group: "latency", render: q => seconds(metric(q, "tool_execution_seconds")), visible: hasValue(q => metric(q, "tool_execution_seconds")) },
        { key: "latency.neo4j_query_seconds", label: "Neo4j Latency", group: "latency", render: q => seconds(metric(q, "neo4j_query_seconds")), visible: hasValue(q => metric(q, "neo4j_query_seconds")) },
        { key: "latency.parsing_validation_seconds", label: "Validation Latency", group: "latency", render: q => seconds(metric(q, "parsing_validation_seconds")), visible: hasValue(q => metric(q, "parsing_validation_seconds")) },
        { key: "latency.retry_wait_seconds", label: "Retry Wait", group: "latency", render: q => seconds(metric(q, "retry_wait_seconds")), visible: hasValue(q => metric(q, "retry_wait_seconds")) },
        { key: "cost", label: "Cost USD", group: "cost", render: q => money(cost(q)), sortValue: q => cost(q), visible: hasValue(q => cost(q)) },
        { key: "local_resources.cpu_avg_percent", label: "CPU Avg", group: "local_resources", render: q => percent(questionLocalSummary(q, "cpu_avg_percent")), sortValue: q => questionLocalSummary(q, "cpu_avg_percent"), visible: hasRecorded(q => questionLocalSummary(q, "cpu_avg_percent")) },
        { key: "local_resources.cpu_peak_percent", label: "CPU Peak", group: "local_resources", render: q => percent(questionLocalSummary(q, "cpu_peak_percent")), sortValue: q => questionLocalSummary(q, "cpu_peak_percent"), visible: hasRecorded(q => questionLocalSummary(q, "cpu_peak_percent")) },
        { key: "local_resources.ram_peak_bytes", label: "RAM Peak", group: "local_resources", render: q => bytes(questionLocalSummary(q, "ram_peak_bytes")), sortValue: q => questionLocalSummary(q, "ram_peak_bytes"), visible: hasRecorded(q => questionLocalSummary(q, "ram_peak_bytes")) },
        { key: "local_resources.gpu_vram_peak_mib", label: "VRAM Peak", group: "local_resources", render: q => mib(questionLocalSummary(q, "gpu_vram_peak_mib")), sortValue: q => questionLocalSummary(q, "gpu_vram_peak_mib"), visible: hasRecorded(q => questionLocalSummary(q, "gpu_vram_peak_mib")) },
        { key: "local_resources.gpu_avg_percent", label: "GPU Avg", group: "local_resources", render: q => percent(questionLocalSummary(q, "gpu_avg_percent")), sortValue: q => questionLocalSummary(q, "gpu_avg_percent"), visible: hasRecorded(q => questionLocalSummary(q, "gpu_avg_percent")) },
        { key: "local_resources.gpu_power_avg_w", label: "GPU Power Avg", group: "local_resources", render: q => watts(questionLocalSummary(q, "gpu_power_avg_w")), sortValue: q => questionLocalSummary(q, "gpu_power_avg_w"), visible: hasRecorded(q => questionLocalSummary(q, "gpu_power_avg_w")) },
        { key: "local_resources.gpu_energy_wh", label: "GPU Energy", group: "local_resources", render: q => wattHours(questionLocalSummary(q, "gpu_energy_wh")), sortValue: q => questionLocalSummary(q, "gpu_energy_wh"), visible: hasRecorded(q => questionLocalSummary(q, "gpu_energy_wh")) },
      ];
      renderTable(
        document.getElementById("questions"),
        columns,
        rowsForActiveMetricGroup(questionsForOverlap(), "questions"),
        q => `data-question-id="${escapeHtml(q.id)}"`
      );
      const questionRows = [...document.querySelectorAll("tr[data-question-id]")];
      questionRows.forEach(row => {
        row.addEventListener("click", () => selectQuestionRow(row));
      });
      const selectedRow = questionRows.find(row => row.dataset.questionId === state.selectedId);
      selectQuestionRow(selectedRow || questionRows[0] || null);
    }
    function selectQuestionRow(row) {
      document.querySelectorAll("tr[data-question-id]").forEach(candidate => {
        candidate.classList.toggle("active", candidate === row);
      });
      state.selectedId = row ? row.dataset.questionId : null;
      renderDetail();
    }
    function prettyValue(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "string") return value;
      try {
        return JSON.stringify(value, null, 2);
      } catch (_error) {
        return text(value);
      }
    }
    function roleLabel(role) {
      return text(role || "Message")
        .split(/[ _-]+/)
        .filter(Boolean)
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }
    function renderMessage(message, index) {
      const content = prettyValue(message.content);
      const isToolResponse = message.role === "tool" || message.kind === "tool_result";
      const isToolCall = !isToolResponse && (
        message.kind === "tool_call" || message.tool_name || message.tool_calls
      );
      const reasoningHtml = message.reasoning
        ? `<h5>Reasoning</h5><pre>${escapeHtml(prettyValue(message.reasoning))}</pre>`
        : "";
      const toolArgsHtml = message.tool_args
        ? `<h5>Tool Arguments</h5><pre>${escapeHtml(prettyValue(message.tool_args))}</pre>`
        : "";
      const toolCallsHtml = message.tool_calls
        ? `<h5>Tool Calls</h5><pre>${escapeHtml(prettyValue(message.tool_calls))}</pre>`
        : "";
      const metadataHtml = message.metadata && !isToolResponse
        ? `<h5>Metadata</h5><pre>${escapeHtml(prettyValue(message.metadata))}</pre>`
        : "";
      const contentHtml = isToolCall ? "" : `<pre>${escapeHtml(content)}</pre>`;
      const rawMessageHtml = text(message.role).toLowerCase().startsWith("assistant") && message.raw_message !== null && message.raw_message !== undefined
        ? `<details><summary>Raw message</summary><pre>${escapeHtml(prettyValue(message.raw_message))}</pre></details>`
        : "";
      const bodyHtml = isToolCall
        ? `${reasoningHtml}${toolArgsHtml}${toolCallsHtml}${metadataHtml}${rawMessageHtml}`
        : `${contentHtml}${reasoningHtml}${toolArgsHtml}${toolCallsHtml}${metadataHtml}${rawMessageHtml}`;
      const toolName = message.tool_name ? ` · ${escapeHtml(message.tool_name)}` : "";
      const messageLabel = message.kind && message.kind !== "assistant_text"
        ? message.kind
        : message.role;
      return `
        <article class="message-card">
          <div class="message-role">${index}. ${escapeHtml(roleLabel(messageLabel))}${toolName}</div>
          ${bodyHtml}
        </article>
      `;
    }
    function renderDetail() {
      const q = report.questions.find(item => item.id === state.selectedId);
      const target = document.getElementById("detail");
      if (!q) {
        target.innerHTML = '<div class="muted">Select a question row to inspect details.</div>';
        return;
      }
      const sequenceHtml = (q.sequences || []).map((sequence, sequenceIndex) => `
        <section class="sequence-group">
          <h4>Sequence ${sequenceIndex + 1}: ${escapeHtml(sequence.description || "Sequence")}</h4>
          <div class="message-list">
            ${(sequence.messages || []).map((message, messageIndex) =>
              renderMessage(message, messageIndex + 1)
            ).join("") || '<div class="muted">No messages recorded.</div>'}
          </div>
        </section>
      `).join("");
      target.innerHTML = `
        <h3>Topic: ${escapeHtml(q.name)}</h3>
        <div class="muted">${escapeHtml(q.question_type || "")} · ${escapeHtml(overlapLabel(q.overlap_class || "unclassified"))}</div>
        <dl class="kv">
          <dt>Configuration</dt><dd>${escapeHtml(q.configuration)}</dd>
          <dt>Provider / Model</dt><dd>${escapeHtml(q.provider_model)}</dd>
          <dt>Tool Executable</dt><dd>${bool(q.tool_executable)}</dd>
          <dt>Cypher Solution / Grounding Match</dt><dd>${bool(q.cypher_solution_match)}</dd>
          <dt>Final Answer Match</dt><dd>${bool(q.final_answer_match)}</dd>
          <dt>Input Tokens</dt><dd>${escapeHtml(q.input_tokens)}</dd>
          <dt>Cached Input Tokens</dt><dd>${escapeHtml(q.cached_input_tokens)}</dd>
          <dt>Output Tokens</dt><dd>${escapeHtml(q.output_tokens)}</dd>
          <dt>Reasoning Tokens</dt><dd>${escapeHtml(q.reasoning_tokens)}</dd>
          <dt>Tool Calls</dt><dd>${escapeHtml(q.n_tool_calls)}</dd>
        </dl>
        <h4>Question</h4><pre>${escapeHtml(q.question)}</pre>
        <h4>Solution</h4><pre>${escapeHtml(q.solution)}</pre>
        <h4>Answer</h4><pre>${escapeHtml(q.answer)}</pre>
        <h4>Generated Cypher</h4><pre>${escapeHtml(prettyValue(q.generated_cypher))}</pre>
        <h4>Cypher Tool Output</h4><pre>${escapeHtml(prettyValue(q.cypher_tool_output))}</pre>
        <h4>Cypher Validation Issues</h4><pre>${escapeHtml(prettyValue(q.cypher_validation_issues || []))}</pre>
        <details class="validation-detail-group">
            <summary>Latency</summary>
            <div class="validation-detail-group-body">
                <pre>${escapeHtml(JSON.stringify(q.latency || {}, null, 2))}</pre>
            </div>
        </details>
        <details class="validation-detail-group">
            <summary>Cost</summary>
            <div class="validation-detail-group-body">
                <pre>${escapeHtml(JSON.stringify(q.cost || {}, null, 2))}</pre>
            </div>
        </details>
        <details class="validation-detail-group">
            <summary>Local Resources</summary>
            <div class="validation-detail-group-body">
                <pre>${escapeHtml(JSON.stringify(q.local_resources || {}, null, 2))}</pre>
            </div>
        </details>
        <details class="validation-detail-group">
          <summary>Messages (${(q.sequences || []).reduce((total, sequence) => total + (sequence.messages || []).length, 0)})</summary>
          <div class="validation-detail-group-body">
            ${sequenceHtml || '<div class="muted">No messages recorded.</div>'}
          </div>
        </details>
      `;
    }
    function selectTab(button) {
      const panelId = button.dataset.tab;
      tabButtons.forEach(candidate => {
        const active = candidate === button;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-selected", active ? "true" : "false");
      });
      tabPanels.forEach(panel => panel.classList.toggle("active", panel.id === panelId));
      metricControls.hidden = !["overview-tab", "questions-tab"].includes(panelId);
    }
    function render() {
      renderOverview();
      renderProviderModels();
      renderQuestions();
      renderArtifacts();
    }
    tabButtons.forEach(button => button.addEventListener("click", () => selectTab(button)));
    groupInputs.forEach(input => input.addEventListener("change", render));
    overlapFilter.addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
