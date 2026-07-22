"""Shared loading, summarization, and display helpers for experiment results."""

from __future__ import annotations

import json
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Sequence

import yaml
from rich.console import Console
from rich.table import Table


TERMINAL_QUESTION_COLUMNS = {
    "Topic": "name",
    "Question": "question",
    "Solution": "solution",
    "Answer": "answer",
    "Completed": "completed",
    "Valid Answer": "valid_answer_format",
    "Correct": "correct",
    "Input Tokens": "input_tokens",
    "Output Tokens": "output_tokens",
    "Tool Calls": "n_tool_calls",
}

TERMINAL_SUMMARY_COLUMNS = {
    "Questions": "questions",
    "Completed": "completed",
    "Valid Answer": "valid_answer_format",
    "Correct": "correct",
    "Input Tokens": "input_tokens",
    "Output Tokens": "output_tokens",
    "Tool Calls": "n_tool_calls",
}


@dataclass
class ProviderModelRef:
    phase: str
    provider: str | None
    model_identifier: str | None


@dataclass
class ConfigurationSummary:
    questions: int = 0
    completed_count: int = 0
    completed_rate: float | None = None
    valid_answer_count: int = 0
    valid_answer_rate: float | None = None
    correct_count: int = 0
    accuracy: float | None = None
    input_tokens_total: int = 0
    input_tokens_avg: float | None = None
    output_tokens_total: int = 0
    output_tokens_avg: float | None = None
    tool_calls_total: int = 0
    tool_calls_avg: float | None = None
    end_to_end_latency_total: float | None = None
    end_to_end_latency_avg: float | None = None
    end_to_end_latency_p50: float | None = None
    end_to_end_latency_p95: float | None = None
    llm_latency_total: float | None = None
    llm_latency_avg: float | None = None
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
    input_tokens: int | None
    output_tokens: int | None
    n_tool_calls: int | None
    latency: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] | None = None
    n_sequences: int = 0
    sequences: list[dict[str, Any]] = field(default_factory=list)
    provider_model: str = ""


@dataclass
class ConfigurationResult:
    source_path: Path
    configuration_name: str
    provider_models: list[ProviderModelRef]
    questions: list[QuestionResult]
    summary: ConfigurationSummary
    cost_summary: dict[str, Any] | None


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
                summary=summarize_configuration(questions),
                cost_summary=_as_dict_or_none(data.get("cost_summary")),
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
                summary=summarize_configuration(questions),
                cost_summary=_as_dict_or_none(configuration_data.get("cost_summary")),
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
    input_tokens = [_number_or_zero(q.input_tokens) for q in questions]
    output_tokens = [_number_or_zero(q.output_tokens) for q in questions]
    tool_calls = [_number_or_zero(q.n_tool_calls) for q in questions]

    e2e = _metric_values(questions, "end_to_end_seconds")
    llm = _metric_values(questions, "llm_call_seconds")
    tool = _metric_values(questions, "tool_execution_seconds")
    neo4j = _metric_values(questions, "neo4j_query_seconds")
    validation = _metric_values(questions, "parsing_validation_seconds")
    retry = _metric_values(questions, "retry_wait_seconds")
    costs = [
        _coerce_float((q.cost or {}).get("total_cost_usd"))
        for q in questions
        if q.cost is not None
    ]
    known_costs = [cost for cost in costs if cost is not None]
    cost_total = round(sum(known_costs), 12) if known_costs else None

    return ConfigurationSummary(
        questions=n_questions,
        completed_count=completed_count,
        completed_rate=completed_count / n_questions,
        valid_answer_count=valid_answer_count,
        valid_answer_rate=valid_answer_count / n_questions,
        correct_count=correct_count,
        accuracy=correct_count / n_questions,
        input_tokens_total=int(sum(input_tokens)),
        input_tokens_avg=sum(input_tokens) / n_questions,
        output_tokens_total=int(sum(output_tokens)),
        output_tokens_avg=sum(output_tokens) / n_questions,
        tool_calls_total=int(sum(tool_calls)),
        tool_calls_avg=sum(tool_calls) / n_questions,
        end_to_end_latency_total=_sum_or_none(e2e),
        end_to_end_latency_avg=_avg_or_none(e2e),
        end_to_end_latency_p50=_percentile(e2e, 50),
        end_to_end_latency_p95=_percentile(e2e, 95),
        llm_latency_total=_sum_or_none(llm),
        llm_latency_avg=_avg_or_none(llm),
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
            console.rule(f"[bold yellow]Configuration: {configuration.configuration_name}")
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
        sequences = analyzed_question.get("sequences") or []
        questions.append(
            QuestionResult(
                source_path=source_path,
                configuration_name=configuration_name,
                name=_as_text(question_data.get("name")),
                question=_as_text(question_data.get("question")),
                solution=_as_text(question_data.get("solution")),
                answer=_as_optional_text(analyzed_question.get("answer")),
                completed=bool(analyzed_question.get("completed", False)),
                valid_answer_format=_as_optional_bool(
                    analysis.get("valid_answer_format")
                ),
                correct=_as_optional_bool(analysis.get("correct")),
                input_tokens=_coerce_int(analysis.get("input_tokens")),
                output_tokens=_coerce_int(analysis.get("output_tokens")),
                n_tool_calls=_coerce_int(analysis.get("n_tool_calls")),
                latency=latency if isinstance(latency, dict) else {},
                cost=cost if isinstance(cost, dict) else None,
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


def _terminal_question_row(question: QuestionResult) -> dict[str, Any]:
    return {
        "name": question.name,
        "question": question.question,
        "solution": question.solution,
        "answer": question.answer,
        "completed": question.completed,
        "valid_answer_format": question.valid_answer_format,
        "correct": question.correct,
        "input_tokens": question.input_tokens,
        "output_tokens": question.output_tokens,
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
        "input_tokens": _total_avg(
            summary.input_tokens_total, summary.input_tokens_avg
        ),
        "output_tokens": _total_avg(
            summary.output_tokens_total, summary.output_tokens_avg
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
        for value in (_coerce_float(question.latency.get(key)) for question in questions)
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
            provider_model_label = _provider_model_label(configuration.provider_models)
            overview.append(
                {
                    "source_id": source_id,
                    "source_path": str(source.path),
                    "configuration": configuration.configuration_name,
                    "provider_model": provider_model_label,
                    "summary": summary,
                    "cost_summary": configuration.cost_summary,
                }
            )
            for ref in configuration.provider_models:
                provider_models.append(
                    {
                        "source_id": source_id,
                        "source_path": str(source.path),
                        "configuration": configuration.configuration_name,
                        "phase": ref.phase,
                        "provider": ref.provider,
                        "model_identifier": ref.model_identifier,
                    }
                )
            for question_index, question in enumerate(configuration.questions):
                questions.append(
                    {
                        "id": f"{config_id}:question-{question_index}",
                        "source_id": source_id,
                        "source_path": str(source.path),
                        "configuration": configuration.configuration_name,
                        "provider_model": question.provider_model,
                        "name": question.name,
                        "question": question.question,
                        "solution": question.solution,
                        "answer": question.answer,
                        "completed": question.completed,
                        "valid_answer_format": question.valid_answer_format,
                        "correct": question.correct,
                        "input_tokens": question.input_tokens,
                        "output_tokens": question.output_tokens,
                        "n_tool_calls": question.n_tool_calls,
                        "latency": question.latency,
                        "cost": question.cost,
                        "n_sequences": question.n_sequences,
                        "sequences": question.sequences,
                    }
                )

    return {
        "sources": source_payload,
        "overview": overview,
        "provider_models": provider_models,
        "questions": questions,
    }


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
      --accent: #1f6feb;
      --good: #117a37;
      --bad: #b42318;
      --warn: #a15c00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 24px 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 { font-size: 22px; margin: 0 0 4px; }
    h2 { font-size: 16px; margin: 24px 0 10px; }
    .muted { color: var(--muted); }
    .wrap { padding: 0 24px 28px; }
    .controls {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 4px;
      min-width: 0;
      font-size: 12px;
      color: var(--muted);
    }
    select, input {
      width: 100%;
      max-width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 6px 8px;
      font: inherit;
    }
    .toggles {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      grid-column: 1 / -1;
      padding-bottom: 3px;
    }
    .toggles label {
      display: inline-flex;
      grid-template-columns: none;
      align-items: center;
      gap: 6px;
      max-width: none;
      white-space: nowrap;
      color: var(--text);
      font-size: 13px;
    }
    .table-wrap {
      max-width: 100%;
      overflow-x: auto;
    }
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
    }
    td {
      max-width: 340px;
      overflow-wrap: anywhere;
    }
    tr[data-question-id] { cursor: pointer; }
    tr[data-question-id]:hover { background: #f2f6ff; }
    .status-true { color: var(--good); font-weight: 600; }
    .status-false { color: var(--bad); font-weight: 600; }
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
      border-radius: 6px;
      padding: 10px 12px 12px;
    }
    .detail {
      position: sticky;
      top: 118px;
      max-height: calc(100vh - 140px);
      overflow: auto;
    }
    .detail h2:first-child { margin-top: 0; }
    .sequence-response { margin-top: 10px; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f2f4f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      max-height: 260px;
      overflow: auto;
    }
    .kv {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 6px 10px;
      margin: 8px 0;
    }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; overflow-wrap: anywhere; }
    .empty {
      padding: 16px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
    }
    @media (max-width: 980px) {
      .controls { grid-template-columns: minmax(0, 1fr); }
      .grid { grid-template-columns: 1fr; }
      .detail { position: static; max-height: none; }
      th { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Heracles Experiment Results</h1>
    <div class="muted">Generated __GENERATED_AT__</div>
    <div class="controls">
      <label>Source <select id="sourceFilter"></select></label>
      <label>Configuration <select id="configFilter"></select></label>
      <label>Provider / Model <select id="modelFilter"></select></label>
      <label>Correctness <select id="correctFilter">
        <option value="all">All</option>
        <option value="correct">Correct</option>
        <option value="incorrect">Incorrect</option>
        <option value="incomplete">Incomplete</option>
      </select></label>
      <label class="search-control">Search <input id="searchInput" type="search" placeholder="Question, solution, answer"></label>
      <div class="toggles">
        <label><input type="checkbox" data-group="quality" checked> Quality</label>
        <label><input type="checkbox" data-group="tokens"> Tokens/tools</label>
        <label><input type="checkbox" data-group="latency"> Latency</label>
        <label><input type="checkbox" data-group="cost"> Cost</label>
      </div>
    </div>
  </header>
  <main class="wrap">
    <h2>Sources</h2>
    <div id="sources"></div>
    <h2>Overview</h2>
    <div id="overview"></div>
    <h2>Provider Models</h2>
    <div id="providerModels"></div>
    <h2>Questions</h2>
    <div class="grid">
      <div id="questions"></div>
      <aside id="detail" class="panel detail">
        <div class="muted">Select a question row to inspect details.</div>
      </aside>
    </div>
  </main>
  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const state = { sortKey: "source_path", sortDir: 1, selectedId: null };
    const filters = {
      source: document.getElementById("sourceFilter"),
      config: document.getElementById("configFilter"),
      model: document.getElementById("modelFilter"),
      correct: document.getElementById("correctFilter"),
      search: document.getElementById("searchInput"),
    };
    const groupInputs = [...document.querySelectorAll("[data-group]")];

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
    function money(value) {
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (!Number.isFinite(n)) return text(value);
      return "$" + n.toFixed(6).replace(/0+$/, "").replace(/\\.$/, "");
    }
    function bool(value) {
      if (value === true) return '<span class="status-true">true</span>';
      if (value === false) return '<span class="status-false">false</span>';
      return '<span class="status-null"></span>';
    }
    function escapeHtml(value) {
      return text(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function options(select, values, allLabel) {
      const current = select.value;
      select.innerHTML = `<option value="all">${allLabel}</option>` +
        values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
      if ([...select.options].some(o => o.value === current)) select.value = current;
    }
    function unique(values) {
      return [...new Set(values.filter(v => v !== null && v !== undefined && v !== ""))].sort();
    }
    function metric(q, key) {
      return q.latency && q.latency[key] !== undefined ? q.latency[key] : null;
    }
    function cost(q) {
      return q.cost && q.cost.total_cost_usd !== undefined ? q.cost.total_cost_usd : null;
    }
    function hasMeaningfulValue(value) {
      if (value === null || value === undefined || value === "") return false;
      if (typeof value === "number") return value !== 0;
      return true;
    }
    function renderTable(target, columns, rows, rowAttrs = () => "") {
      const visibleColumns = columns.filter(c => !c.visible || rows.some(row => c.visible(row)));
      const sortedRows = sortRows(rows, visibleColumns);
      if (!sortedRows.length) {
        target.innerHTML = '<div class="empty">No rows match the current filters.</div>';
        return;
      }
      const header = visibleColumns.map(c => `<th data-sort="${escapeHtml(c.key)}" data-group="${escapeHtml(c.group || "")}">${escapeHtml(c.label)}</th>`).join("");
      const body = sortedRows.map(row => {
        const cells = visibleColumns.map(c => `<td data-group="${escapeHtml(c.group || "")}">${c.render ? c.render(row) : escapeHtml(row[c.key])}</td>`).join("");
        return `<tr ${rowAttrs(row)}>${cells}</tr>`;
      }).join("");
      target.innerHTML = `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
      target.querySelectorAll("th[data-sort]").forEach(th => {
        th.addEventListener("click", () => {
          const key = th.dataset.sort;
          state.sortDir = state.sortKey === key ? -state.sortDir : 1;
          state.sortKey = key;
          render();
        });
      });
    }
    function sortRows(rows, columns) {
      const column = columns.find(c => c.key === state.sortKey);
      if (!column) return [...rows];
      return [...rows].sort((a, b) => {
        const av = column.sortValue ? column.sortValue(a) : valueByPath(a, column.key);
        const bv = column.sortValue ? column.sortValue(b) : valueByPath(b, column.key);
        return String(av ?? "").localeCompare(String(bv ?? ""), undefined, { numeric: true }) * state.sortDir;
      });
    }
    function valueByPath(row, path) {
      return path.split(".").reduce((value, key) => value && value[key], row);
    }
    function renderSources() {
      renderTable(document.getElementById("sources"), [
        { key: "path", label: "Path" },
        { key: "source_experiment", label: "Source Experiment", render: s => escapeHtml(s.metadata.source_experiment), sortValue: s => s.metadata.source_experiment },
        { key: "elapsed_seconds", label: "Elapsed Seconds", render: s => number(s.metadata.elapsed_seconds), sortValue: s => s.metadata.elapsed_seconds },
        { key: "failed_configurations", label: "Failed Configurations", render: s => escapeHtml(JSON.stringify(s.metadata.failed_configurations || {})), sortValue: s => JSON.stringify(s.metadata.failed_configurations || {}) },
      ], report.sources);
    }
    function renderOverview() {
      renderTable(document.getElementById("overview"), [
        { key: "source_path", label: "Source" },
        { key: "configuration", label: "Configuration" },
        { key: "provider_model", label: "Provider / Model" },
        { key: "questions", label: "Questions", render: r => text(r.summary.questions), sortValue: r => r.summary.questions },
        { key: "accuracy", label: "Accuracy", render: r => number((r.summary.accuracy || 0) * 100, 1) + "%", sortValue: r => r.summary.accuracy },
        { key: "completed_rate", label: "Completed", render: r => number((r.summary.completed_rate || 0) * 100, 1) + "%", sortValue: r => r.summary.completed_rate },
        { key: "input_tokens_total", label: "Input Tokens", render: r => text(r.summary.input_tokens_total), sortValue: r => r.summary.input_tokens_total },
        { key: "output_tokens_total", label: "Output Tokens", render: r => text(r.summary.output_tokens_total), sortValue: r => r.summary.output_tokens_total },
        { key: "end_to_end_latency_avg", label: "Avg E2E Latency (s)", render: r => number(r.summary.end_to_end_latency_avg), sortValue: r => r.summary.end_to_end_latency_avg },
        { key: "cost_total_usd", label: "Cost", render: r => money(r.summary.cost_total_usd), sortValue: r => r.summary.cost_total_usd },
      ], report.overview);
    }
    function renderProviderModels() {
      renderTable(document.getElementById("providerModels"), [
        { key: "source_path", label: "Source" },
        { key: "configuration", label: "Configuration" },
        { key: "phase", label: "Phase" },
        { key: "provider", label: "Provider" },
        { key: "model_identifier", label: "Model Identifier" },
      ], report.provider_models);
    }
    function filteredQuestions() {
      const term = filters.search.value.trim().toLowerCase();
      return report.questions.filter(q => {
        if (filters.source.value !== "all" && q.source_path !== filters.source.value) return false;
        if (filters.config.value !== "all" && q.configuration !== filters.config.value) return false;
        if (filters.model.value !== "all" && q.provider_model !== filters.model.value) return false;
        if (filters.correct.value === "correct" && q.correct !== true) return false;
        if (filters.correct.value === "incorrect" && q.correct !== false) return false;
        if (filters.correct.value === "incomplete" && q.completed === true) return false;
        if (!term) return true;
        return [q.name, q.question, q.solution, q.answer].some(v => text(v).toLowerCase().includes(term));
      });
    }
    function renderQuestions() {
      const rows = filteredQuestions();
      const hasValue = accessor => row => hasMeaningfulValue(accessor(row));
      const columns = [
        { key: "source_path", label: "Source" },
        { key: "configuration", label: "Configuration" },
        { key: "provider_model", label: "Provider / Model" },
        { key: "name", label: "Topic" },
        { key: "completed", label: "Completed", group: "quality", render: q => bool(q.completed) },
        { key: "correct", label: "Correct", group: "quality", render: q => bool(q.correct) },
        { key: "valid_answer_format", label: "Valid Answer", group: "quality", render: q => bool(q.valid_answer_format) },
        { key: "input_tokens", label: "Input Tokens", group: "tokens", visible: hasValue(q => q.input_tokens) },
        { key: "output_tokens", label: "Output Tokens", group: "tokens", visible: hasValue(q => q.output_tokens) },
        { key: "n_tool_calls", label: "Tool Calls", group: "tokens", visible: hasValue(q => q.n_tool_calls) },
        { key: "latency.end_to_end_seconds", label: "End-to-End Latency", group: "latency", render: q => seconds(metric(q, "end_to_end_seconds")), visible: hasValue(q => metric(q, "end_to_end_seconds")) },
        { key: "latency.llm_call_seconds", label: "LLM Latency", group: "latency", render: q => seconds(metric(q, "llm_call_seconds")), visible: hasValue(q => metric(q, "llm_call_seconds")) },
        { key: "latency.tool_execution_seconds", label: "Tool Latency", group: "latency", render: q => seconds(metric(q, "tool_execution_seconds")), visible: hasValue(q => metric(q, "tool_execution_seconds")) },
        { key: "latency.neo4j_query_seconds", label: "Neo4j Latency", group: "latency", render: q => seconds(metric(q, "neo4j_query_seconds")), visible: hasValue(q => metric(q, "neo4j_query_seconds")) },
        { key: "latency.parsing_validation_seconds", label: "Validation Latency", group: "latency", render: q => seconds(metric(q, "parsing_validation_seconds")), visible: hasValue(q => metric(q, "parsing_validation_seconds")) },
        { key: "latency.retry_wait_seconds", label: "Retry Wait", group: "latency", render: q => seconds(metric(q, "retry_wait_seconds")), visible: hasValue(q => metric(q, "retry_wait_seconds")) },
        { key: "latency.time_to_first_token_seconds", label: "Time to First Token", group: "latency", render: q => seconds(metric(q, "time_to_first_token_seconds")), visible: hasValue(q => metric(q, "time_to_first_token_seconds")) },
        { key: "cost", label: "Cost USD", group: "cost", render: q => money(cost(q)), sortValue: q => cost(q), visible: hasValue(q => cost(q)) },
      ];
      renderTable(
        document.getElementById("questions"),
        columns,
        rows,
        q => `data-question-id="${escapeHtml(q.id)}"`
      );
      document.querySelectorAll("tr[data-question-id]").forEach(row => {
        row.addEventListener("click", () => {
          state.selectedId = row.dataset.questionId;
          renderDetail();
        });
      });
      applyGroupVisibility();
      if (!rows.some(q => q.id === state.selectedId)) state.selectedId = rows[0] && rows[0].id;
      renderDetail();
    }
    function renderDetail() {
      const q = report.questions.find(item => item.id === state.selectedId);
      const target = document.getElementById("detail");
      if (!q) {
        target.innerHTML = '<div class="muted">Select a question row to inspect details.</div>';
        return;
      }
      const sequenceHtml = (q.sequences || []).map(seq => `
        <h2>${escapeHtml(seq.description || "Sequence")}</h2>
        ${(seq.responses || []).map((r, i) => `
          <div class="panel sequence-response">
            <strong>Response ${i + 1}</strong>
            <pre>${escapeHtml(r.raw_response)}</pre>
          </div>
        `).join("")}
      `).join("");
      target.innerHTML = `
        <h2>Topic: ${escapeHtml(q.name)}</h2>
        <dl class="kv">
          <dt>Source</dt><dd>${escapeHtml(q.source_path)}</dd>
          <dt>Configuration</dt><dd>${escapeHtml(q.configuration)}</dd>
          <dt>Provider / Model</dt><dd>${escapeHtml(q.provider_model)}</dd>
          <dt>Completed</dt><dd>${bool(q.completed)}</dd>
          <dt>Correct</dt><dd>${bool(q.correct)}</dd>
          <dt>Valid Answer</dt><dd>${bool(q.valid_answer_format)}</dd>
          <dt>Input Tokens</dt><dd>${escapeHtml(q.input_tokens)}</dd>
          <dt>Output Tokens</dt><dd>${escapeHtml(q.output_tokens)}</dd>
          <dt>Tool Calls</dt><dd>${escapeHtml(q.n_tool_calls)}</dd>
        </dl>
        <h2>Question</h2><pre>${escapeHtml(q.question)}</pre>
        <h2>Solution</h2><pre>${escapeHtml(q.solution)}</pre>
        <h2>Answer</h2><pre>${escapeHtml(q.answer)}</pre>
        <h2>Latency (s)</h2><pre>${escapeHtml(JSON.stringify(q.latency || {}, null, 2))}</pre>
        <h2>Cost</h2><pre>${escapeHtml(JSON.stringify(q.cost || {}, null, 2))}</pre>
        <h2>Sequences</h2>${sequenceHtml || '<div class="muted">No sequences recorded.</div>'}
      `;
    }
    function setExclusiveGroupSelection(changedInput) {
      groupInputs.forEach(input => {
        input.checked = input === changedInput;
      });
    }
    function applyGroupVisibility() {
      const active = new Map(groupInputs.map(input => [input.dataset.group, input.checked]));
      document.querySelectorAll("th[data-group], td[data-group]").forEach(el => {
        const group = el.dataset.group;
        if (!group) return;
        el.style.display = active.get(group) ? "" : "none";
      });
    }
    function populateFilters() {
      options(filters.source, unique(report.questions.map(q => q.source_path)), "All sources");
      options(filters.config, unique(report.questions.map(q => q.configuration)), "All configurations");
      options(filters.model, unique(report.questions.map(q => q.provider_model)), "All provider/models");
    }
    function render() {
      renderSources();
      renderOverview();
      renderProviderModels();
      renderQuestions();
    }
    populateFilters();
    Object.values(filters).forEach(el => el.addEventListener("input", render));
    groupInputs.forEach(el => el.addEventListener("change", () => {
      setExclusiveGroupSelection(el);
      applyGroupVisibility();
    }));
    render();
  </script>
</body>
</html>
"""
