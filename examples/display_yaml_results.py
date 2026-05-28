#!/usr/bin/env python3
"""Display Heracles experiment result YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table


DEFAULT_COLUMNS = {
    "Name": "name",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Heracles experiment result YAML as rich tables.",
    )
    parser.add_argument("results_yaml", type=Path, help="Result YAML file to display.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=96,
        help="Maximum displayed width for long cell values. Defaults to 96.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary tables.",
    )
    parser.add_argument(
        "--show-sequences",
        action="store_true",
        help="Add a sequence count column when sequence data is present.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as fo:
        data = yaml.safe_load(fo)
    if not isinstance(data, dict):
        raise ValueError(f"Result YAML must contain a mapping: {path}")
    return data


def colorize_bool(value: bool) -> str:
    color = "green" if value else "red"
    return f"[{color}]{value}[/{color}]"


def format_cell(value: Any, max_width: int) -> str:
    if isinstance(value, bool):
        return colorize_bool(value)
    if value is None:
        return ""
    text = str(value)
    if max_width > 0 and len(text) > max_width:
        return text[: max_width - 1] + "..."
    return text


def construct_question_dict(analyzed_question: dict[str, Any]) -> dict[str, Any]:
    analysis = analyzed_question.get("analysis") or {}
    question = analyzed_question.get("question") or {}

    return {
        "name": question.get("name", ""),
        "question": question.get("question", ""),
        "solution": question.get("solution", ""),
        "answer": analyzed_question.get("answer", ""),
        "completed": analyzed_question.get("completed", ""),
        "valid_answer_format": analysis.get("valid_answer_format", ""),
        "correct": analysis.get("correct", ""),
        "input_tokens": analysis.get("input_tokens", 0),
        "output_tokens": analysis.get("output_tokens", 0),
        "n_tool_calls": analysis.get("n_tool_calls", 0),
        "n_sequences": len(analyzed_question.get("sequences") or []),
    }


def summarize_results(questions: list[dict[str, Any]]) -> dict[str, Any]:
    n_questions = len(questions)
    if n_questions == 0:
        return {"questions": 0}

    summary = {"questions": n_questions}
    for key in ("completed", "valid_answer_format", "correct"):
        count = sum(1 for q in questions if q.get(key) is True)
        pct = 100 * count / n_questions
        summary[key] = f"{count}/{n_questions} ({pct:.1f}%)"

    for key in ("input_tokens", "output_tokens", "n_tool_calls"):
        total = sum(q.get(key) or 0 for q in questions)
        summary[key] = f"{total} total, {total / n_questions:.1f} avg"

    return summary


def make_table(
    title: str,
    rows: list[dict[str, Any]],
    columns: dict[str, str],
    *,
    max_width: int,
) -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for column_name in columns:
        table.add_column(column_name, overflow="fold")

    if not rows:
        table.add_row(*[""] * len(columns))
        return table

    for row in rows:
        table.add_row(
            *[format_cell(row.get(key, ""), max_width) for key in columns.values()]
        )
    return table


def result_configurations(results: dict[str, Any]) -> dict[str, Any]:
    if "experiment_configurations" in results:
        return results["experiment_configurations"] or {}
    if "analyzed_questions" in results:
        return {"results": results}
    raise ValueError(
        "Result YAML must contain either 'experiment_configurations' or "
        "'analyzed_questions'."
    )


def display_results(
    results: dict[str, Any],
    *,
    console: Console,
    max_width: int,
    summary_only: bool,
    show_sequences: bool,
) -> None:
    configurations = result_configurations(results)
    if not configurations:
        console.print("[yellow]No experiment configurations found.[/yellow]")
        return

    columns = dict(DEFAULT_COLUMNS)
    if show_sequences:
        columns["Sequences"] = "n_sequences"

    for config_name, config_data in configurations.items():
        console.rule(f"[bold yellow]Configuration: {config_name}")
        analyzed_questions = config_data.get("analyzed_questions") or []
        questions = [construct_question_dict(q) for q in analyzed_questions]

        if not summary_only:
            console.print(
                make_table(
                    "Per-Question Results",
                    questions,
                    columns,
                    max_width=max_width,
                )
            )

        summary = summarize_results(questions)
        console.print(
            make_table(
                "Summary",
                [summary],
                {key.replace("_", " ").title(): key for key in summary},
                max_width=max_width,
            )
        )


def main() -> int:
    args = parse_args()
    console = Console()
    try:
        results = load_yaml(args.results_yaml.expanduser())
        display_results(
            results,
            console=console,
            max_width=args.max_width,
            summary_only=args.summary_only,
            show_sequences=args.show_sequences,
        )
    except Exception as ex:
        console.print(f"[red]Error:[/red] {ex}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
