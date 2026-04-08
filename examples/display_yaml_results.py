#!/usr/bin/env python3

import sys
import os
import yaml
from rich.console import Console
from rich.table import Table

# -------- Helper Functions --------
def to_string(value):
    if isinstance(value, bool):
        return f"[{'green' if value else 'red'}]{value}[/{'green' if value else 'red'}]"
    return str(value)

def generate_table(title, row_data, column_data_map):
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col_name in column_data_map:
        table.add_column(col_name)

    for i, row in enumerate(row_data):
        table.add_row(*[to_string(row.get(col_key, "")) for col_key in column_data_map.values()])
        table.add_row(*[""] * len(column_data_map)) # Add empty row(s) to simulate spacing
    return table

def display_table(title, row_data, column_data_map):
    table = generate_table(title, row_data, column_data_map)
    console = Console()
    console.print(table)

def construct_question_dict(aq):
    """
    Convert an analyzed_question into a flat dictionary for display.
    """
    analysis = aq.get("analysis", {})
    question = aq.get("question", {})

    return {
        "name": question.get("name", ""),
        "question": question.get("question", ""),
        "solution": question.get("solution", ""),
        "answer": aq.get("answer", ""),
        "valid_answer_format": analysis.get("valid_answer_format", ""),
        "correct": analysis.get("correct", ""),
        "input_tokens": analysis.get("input_tokens", ""),
        "output_tokens": analysis.get("output_tokens", ""),
        "n_tool_calls": analysis.get("n_tool_calls", ""),
    }

def summarize_results(questions):
    """
    Summarize numeric/boolean fields over all questions.
    """
    n_questions = len(questions)
    if n_questions == 0:
        return {}

    acc = {
        "valid_answer_format": 0,
        "correct": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "n_tool_calls": 0,
    }

    for q in questions:
        for k in acc:
            val = q.get(k, 0)
            if isinstance(val, bool):
                acc[k] += int(val)
            else:
                acc[k] += val

    summary = {k: f"{v}/{n_questions}" for k, v in acc.items()}
    summary["questions"] = n_questions
    return summary

# -------- Main Script --------
def main():
    if len(sys.argv) < 2:
        print("Usage: ./read_experiment_results.py path_to_results_yaml")
        sys.exit(1)

    yaml_file = sys.argv[1]

    if not os.path.isfile(yaml_file):
        print(f"Error: File not found: {yaml_file}")
        sys.exit(1)

    with open(yaml_file, "r") as fo:
        results = yaml.safe_load(fo)

    console = Console()

    experiment_configs = results.get("experiment_configurations", {})

    for config_name, config_data in experiment_configs.items():
        console.rule(f"[bold yellow]Configuration: {config_name}")

        analyzed_questions = config_data.get("analyzed_questions", [])
        questions = [construct_question_dict(q) for q in analyzed_questions]

        # Per-question table
        column_data_map = {
            "Name": "name",
            "Question": "question",
            "Solution": "solution",
            "Answer": "answer",
            "Valid Answer": "valid_answer_format",
            "Correct": "correct",
            "Input Tokens": "input_tokens",
            "Output Tokens": "output_tokens",
            "Tool Calls": "n_tool_calls",
        }
        display_table("Per-Question Results", questions, column_data_map)

        # Summary table
        summary = [summarize_results(questions)]
        summary_column_data_map = {k: k for k in summary[0].keys()}
        display_table("Summary", summary, summary_column_data_map)

if __name__ == "__main__":
    main()