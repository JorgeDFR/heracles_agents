#!/usr/bin/env python3
"""Run one or more Heracles experiment YAML files."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from time import perf_counter

import yaml

from heracles_agents.cli.summarize_results import display_experiment_results
from heracles_agents.experiment_definition import ExperimentDescription
from heracles_agents.llm_interface import AnalyzedExperiment


logger = logging.getLogger(__name__)
EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Heracles experiment configuration files.",
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help=(
            "Experiment YAML files. Relative paths are resolved from the current "
            "directory first, then from examples/."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=REPO_ROOT / "output",
        type=Path,
        help="Directory for result YAML files. Defaults to ./output.",
    )
    parser.add_argument(
        "-c",
        "--configuration",
        action="append",
        dest="configurations",
        help=(
            "Run only this configuration name. May be passed multiple times. "
            "By default all configurations are run."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later configurations/experiments after a failure.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not print per-configuration result tables while running.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def resolve_experiment_path(value: str) -> Path:
    candidates = [
        Path(value).expanduser(),
        EXAMPLES_DIR / value,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Experiment file not found. Checked: {checked}")


def load_experiment(path: Path) -> ExperimentDescription:
    with path.open("r", encoding="utf-8") as fo:
        data = yaml.safe_load(fo)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment file must contain a YAML mapping: {path}")
    return ExperimentDescription(**data)


def output_path_for(experiment_path: Path, output_dir: Path) -> Path:
    try:
        relative_path = experiment_path.relative_to(EXAMPLES_DIR / "experiments")
        provider_dir = relative_path.parent
    except ValueError:
        provider_dir = Path(experiment_path.parent.name)

    experiment_name = experiment_path.stem.replace(" ", "_").lower()
    return output_dir / provider_dir / f"{experiment_name}_results.yaml"


def run_experiment(
    experiment_path: Path,
    output_dir: Path,
    selected_configurations: set[str] | None,
    *,
    continue_on_error: bool,
    display_results: bool,
) -> Path:
    logger.info("Running experiment: %s", experiment_path)
    experiment = load_experiment(experiment_path)

    unknown_configurations = (
        selected_configurations - set(experiment.configurations)
        if selected_configurations
        else set()
    )
    if unknown_configurations:
        available = ", ".join(sorted(experiment.configurations))
        requested = ", ".join(sorted(unknown_configurations))
        raise ValueError(
            f"Unknown configuration(s) for {experiment_path}: {requested}. "
            f"Available: {available}"
        )

    results = {}
    failures = {}
    started_at = perf_counter()
    for configuration_name, experiment_config in experiment.configurations.items():
        if selected_configurations and configuration_name not in selected_configurations:
            continue

        logger.info("Running configuration: %s", configuration_name)
        try:
            analyzed_questions = experiment_config.pipeline.function(experiment_config)
        except Exception as ex:
            logger.exception("Configuration failed: %s", configuration_name)
            failures[configuration_name] = str(ex)
            if continue_on_error:
                continue
            raise

        if display_results:
            display_experiment_results(analyzed_questions, title=configuration_name)
        results[configuration_name] = analyzed_questions

    if not results:
        raise RuntimeError(f"No configurations completed for {experiment_path}")

    elapsed_s = perf_counter() - started_at
    analyzed_experiment = AnalyzedExperiment(
        experiment_configurations=results,
        metadata={
            **experiment.metadata,
            "source_experiment": str(experiment_path),
            "elapsed_seconds": round(elapsed_s, 3),
            "failed_configurations": failures,
        },
    )

    result_path = output_path_for(experiment_path, output_dir)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as fo:
        yaml.safe_dump(analyzed_experiment.model_dump(mode="json"), fo, sort_keys=False)

    logger.info("Saved results: %s", result_path)
    return result_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), force=True)

    os.environ.setdefault("HERACLES_AGENTS_PATH", str(REPO_ROOT))

    selected_configurations = (
        set(args.configurations) if args.configurations is not None else None
    )

    result_paths = []
    for experiment_arg in args.experiments:
        try:
            experiment_path = resolve_experiment_path(experiment_arg)
            result_paths.append(
                run_experiment(
                    experiment_path,
                    args.output_dir.expanduser().resolve(),
                    selected_configurations,
                    continue_on_error=args.continue_on_error,
                    display_results=not args.no_display,
                )
            )
        except Exception:
            logger.exception("Experiment failed: %s", experiment_arg)
            if not args.continue_on_error:
                return 1

    if result_paths:
        logger.info("Completed %d experiment file(s).", len(result_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
