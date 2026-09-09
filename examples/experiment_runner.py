#!/usr/bin/env python3
"""Run one or more Heracles experiment YAML files."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from time import perf_counter

import yaml

from heracles_agents.cli.model_sweeps import expand_model_sweeps
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
        "--debug",
        action="store_true",
        help=(
            "Enable debug artifacts for experiment runs. Currently records raw "
            "local resource samples next to the result YAML when local metrics "
            "are enabled."
        ),
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


def load_experiment_with_context(path: Path) -> tuple[ExperimentDescription, dict]:
    os.environ.setdefault("HERACLES_AGENTS_PATH", str(REPO_ROOT))
    with path.open("r", encoding="utf-8") as fo:
        data = yaml.safe_load(fo)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment file must contain a YAML mapping: {path}")
    data, sweep_context = expand_model_sweeps(data, path)
    return ExperimentDescription(**data), sweep_context


def load_experiment(path: Path) -> ExperimentDescription:
    experiment, _ = load_experiment_with_context(path)
    return experiment


def output_path_for(experiment_path: Path, output_dir: Path) -> Path:
    try:
        relative_path = experiment_path.relative_to(EXAMPLES_DIR / "experiments")
        provider_dir = relative_path.parent
    except ValueError:
        provider_dir = Path(experiment_path.parent.name)

    experiment_name = experiment_path.stem.replace(" ", "_").lower()
    return output_dir / provider_dir / f"{experiment_name}_results.yaml"


def output_path_for_configuration(
    experiment_path: Path,
    output_dir: Path,
    configuration_name: str,
) -> Path:
    try:
        relative_path = experiment_path.relative_to(EXAMPLES_DIR / "experiments")
        provider_dir = relative_path.parent
    except ValueError:
        provider_dir = Path(experiment_path.parent.name)

    experiment_name = experiment_path.stem.replace(" ", "_").lower()
    safe_configuration_name = configuration_name.replace(" ", "_").lower()
    return (
        output_dir
        / provider_dir
        / experiment_name
        / f"{safe_configuration_name}_results.yaml"
    )


def debug_samples_path_for(result_path: Path, configuration_name: str) -> Path:
    safe_configuration_name = configuration_name.replace(" ", "_").lower()
    result_stem = result_path.stem
    if result_stem.endswith("_results"):
        result_stem = result_stem[: -len("_results")]
    return result_path.with_name(
        f"{result_stem}_{safe_configuration_name}_local_metrics_samples.yaml"
    )


def configure_debug_outputs(
    experiment_config,
    result_path: Path,
    configuration_name: str,
) -> None:
    local_metrics = getattr(experiment_config, "local_metrics", None)
    if not isinstance(local_metrics, dict):
        return
    experiment_config.local_metrics = {
        **local_metrics,
        "record_samples": True,
        "samples_output_path": str(
            debug_samples_path_for(result_path, configuration_name)
        ),
    }


def build_llm_metadata(
    experiment: ExperimentDescription,
    configuration_names: set[str],
    result_configuration_names: dict[str, str] | None = None,
) -> dict[str, dict]:
    result_configuration_names = result_configuration_names or {}
    llm_configurations = {}
    for configuration_name, experiment_config in experiment.configurations.items():
        if configuration_name not in configuration_names:
            continue
        result_configuration_name = result_configuration_names.get(
            configuration_name,
            configuration_name,
        )
        phases = {}
        for phase_name, agent in experiment_config.phases.items():
            phases[phase_name] = {
                "provider": agent.client.client_type,
                "model_identifier": agent.model_info.model,
            }
        llm_configurations[result_configuration_name] = {"phases": phases}
    return llm_configurations


def run_experiment(
    experiment_path: Path,
    output_dir: Path,
    selected_configurations: set[str] | None,
    *,
    continue_on_error: bool,
    display_results: bool,
    debug: bool = False,
) -> list[Path]:
    logger.info("Running experiment: %s", experiment_path)
    experiment, sweep_context = load_experiment_with_context(experiment_path)
    result_configuration_names = sweep_context.get("result_configuration_names", {})
    has_model_sweeps = bool(result_configuration_names)

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
        logger.info(
            "Question sweep size: %d",
            len(getattr(experiment_config, "questions", [])),
        )
        if debug:
            result_path = (
                output_path_for_configuration(
                    experiment_path,
                    output_dir,
                    configuration_name,
                )
                if has_model_sweeps
                else output_path_for(experiment_path, output_dir)
            )
            configure_debug_outputs(experiment_config, result_path, configuration_name)
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
    metadata = {
        **experiment.metadata,
        "elapsed_seconds": round(elapsed_s, 3),
        "failed_configurations": failures,
    }
    if "benchmark_manifest" not in metadata:
        metadata["source_experiment"] = str(experiment_path)
    if not has_model_sweeps:
        metadata["llm_configurations"] = build_llm_metadata(experiment, set(results))

    result_paths = []
    if has_model_sweeps:
        for configuration_name, analyzed_questions in results.items():
            result_configuration_name = result_configuration_names.get(
                configuration_name,
                configuration_name,
            )
            analyzed_experiment = AnalyzedExperiment(
                metadata={
                    **metadata,
                    "llm_configurations": build_llm_metadata(
                        experiment,
                        {configuration_name},
                        result_configuration_names,
                    ),
                },
                experiment_configurations={
                    result_configuration_name: analyzed_questions
                },
            )
            result_path = output_path_for_configuration(
                experiment_path,
                output_dir,
                configuration_name,
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w", encoding="utf-8") as fo:
                yaml.safe_dump(
                    analyzed_experiment.model_dump(mode="json"),
                    fo,
                    sort_keys=False,
                )
            logger.info("Saved results: %s", result_path)
            result_paths.append(result_path)
        return result_paths

    analyzed_experiment = AnalyzedExperiment(
        metadata=metadata,
        experiment_configurations=results,
    )
    result_path = output_path_for(experiment_path, output_dir)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as fo:
        yaml.safe_dump(analyzed_experiment.model_dump(mode="json"), fo, sort_keys=False)

    logger.info("Saved results: %s", result_path)
    return [result_path]


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
            result_paths.extend(
                run_experiment(
                    experiment_path,
                    args.output_dir.expanduser().resolve(),
                    selected_configurations,
                    continue_on_error=args.continue_on_error,
                    display_results=not args.no_display,
                    debug=args.debug,
                )
            )
        except Exception:
            logger.exception("Experiment failed: %s", experiment_arg)
            if not args.continue_on_error:
                return 1

    if result_paths:
        logger.info("Wrote %d result file(s).", len(result_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
