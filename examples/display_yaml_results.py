#!/usr/bin/env python3
"""Display Heracles experiment result YAML files."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_reporting_functions():
    # Importing heracles_agents registers tools and currently prints to stdout.
    # Keep this inspection CLI quiet without changing package initialization.
    with contextlib.redirect_stdout(io.StringIO()):
        from heracles_agents.cli.result_reporting import (
            load_result_files,
            render_html_report,
            render_terminal_summary,
        )

    return load_result_files, render_html_report, render_terminal_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Heracles experiment result YAML files.",
    )
    parser.add_argument(
        "results_yaml",
        nargs="+",
        type=Path,
        help="One or more result YAML files to display.",
    )
    parser.add_argument(
        "--mode",
        choices=["terminal", "html"],
        default="terminal",
        help="Display mode. Defaults to terminal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "experiment_results_report.html",
        help=(
            "Output path for --mode html. Defaults to "
            "./output/experiment_results_report.html."
        ),
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_report",
        help="Open the generated HTML report in the default browser.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=96,
        help="Maximum displayed width for long terminal cell values. Defaults to 96.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary tables in terminal mode.",
    )
    parser.add_argument(
        "--show-sequences",
        action="store_true",
        help="Add a sequence count column in terminal mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    try:
        load_result_files, render_html_report, render_terminal_summary = (
            load_reporting_functions()
        )
        sources = load_result_files(args.results_yaml)
        if args.mode == "html":
            output_path = render_html_report(
                sources,
                args.output,
                open_report=args.open_report,
            )
            console.print(f"[green]Saved HTML report:[/green] {output_path}")
        else:
            render_terminal_summary(
                sources,
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
