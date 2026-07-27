#!/usr/bin/env python3
"""External validation monitor for Ollama local resource metrics.

This script is intentionally isolated under examples/. It is a validation
wrapper around an experiment command, not part of the production metrics path.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

PACKAGE_SRC = Path(__file__).resolve().parents[2] / "src"
if PACKAGE_SRC.exists() and str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from heracles_agents.local_metrics.docker_proxy import (
    DockerContainerStatsSample,
    DockerProxyClient,
    calculate_cpu_percent,
)
from heracles_agents.local_metrics.gpu_stats import (
    GpuStatsSample,
    sample_gpu_stats,
)
from heracles_agents.local_metrics.monitor import (
    LocalMetricsConfig,
    LocalResourceMonitor,
    config_from_mapping,
)


TIME_SERIES_FIELDS = [
    "source",
    "sample_kind",
    "timestamp_monotonic",
    "relative_seconds",
    "gpu_index",
    "gpu_name",
    "container_cpu_percent",
    "container_cpu_percent_normalized",
    "container_ram_bytes",
    "container_process_rss_bytes",
    "container_process_vsz_bytes",
    "container_process_count",
    "container_ram_cgroup_working_set_bytes",
    "container_ram_including_cache_bytes",
    "container_ram_file_bytes",
    "container_ram_active_file_bytes",
    "container_ram_inactive_file_bytes",
    "container_ram_anon_bytes",
    "container_ram_limit_bytes",
    "container_pids",
    "gpu_memory_used_mib_raw",
    "gpu_memory_used_mib_adjusted",
    "gpu_utilization_percent_raw",
    "gpu_utilization_percent_adjusted",
    "gpu_memory_utilization_percent",
    "gpu_power_w_raw",
    "gpu_power_w_adjusted",
    "gpu_energy_wh_adjusted_cumulative",
]

EVENT_STYLES = {
    "baseline": {"color": "#7c3aed", "linestyle": "--"},
    "configuration_baseline": {"color": "#0891b2", "linestyle": "--"},
    "external": {"color": "#111827", "linestyle": "-"},
    "pipeline": {"color": "#dc2626", "linestyle": "-"},
    "configuration": {"color": "#2563eb", "linestyle": ":"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command while externally sampling Ollama Docker/GPU resource "
            "metrics for validation."
        )
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Optional experiment YAML used to read metadata.local_metrics.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for validation artifacts.",
    )
    parser.add_argument(
        "--pre-pipeline-seconds",
        type=float,
        default=0.0,
        help="Seconds to sample before launching the pipeline command.",
    )
    parser.add_argument(
        "--post-pipeline-seconds",
        type=float,
        default=0.0,
        help="Seconds to sample after the pipeline command exits.",
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=Path,
        help="Optional output tree to search for internal debug sample files.",
    )
    parser.add_argument(
        "--internal-samples",
        action="append",
        type=Path,
        default=[],
        help=(
            "Internal *_local_metrics_samples.yaml file to use for configuration "
            "event markers. Repeatable."
        ),
    )
    parser.add_argument("--sample-interval-seconds", type=float)
    parser.add_argument("--gpu-baseline-seconds", type=float)
    parser.add_argument("--ollama-container-name")
    parser.add_argument("--docker-host")
    parser.add_argument("--ollama-host")
    parser.add_argument(
        "--no-unload-models-before-baseline",
        action="store_true",
        help="Do not unload currently loaded Ollama models before baseline sampling.",
    )
    parser.add_argument(
        "--no-baseline-adjust-gpu",
        action="store_true",
        help=(
            "Deprecated; external validation plots always use raw GPU values "
            "without baseline adjustment."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip external interface checks before sampling.",
    )
    parser.add_argument(
        "--plots-format",
        choices=["png", "svg"],
        default="png",
        help="Plot file format.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Pipeline command to run after --.",
    )
    args = parser.parse_args(argv)
    if args.pre_pipeline_seconds < 0 or args.post_pipeline_seconds < 0:
        parser.error("pre/post pipeline seconds must be non-negative.")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("pipeline command is required after --.")
    return args


def load_experiment_local_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    metadata = data.get("metadata") if isinstance(data, dict) else {}
    local_metrics = metadata.get("local_metrics") if isinstance(metadata, dict) else {}
    return local_metrics if isinstance(local_metrics, dict) else {}


def build_config(args: argparse.Namespace) -> LocalMetricsConfig:
    config = config_from_mapping(load_experiment_local_metrics(args.experiment))
    config.enabled = True
    config.baseline_adjust_gpu = False
    if args.sample_interval_seconds is not None:
        config.sample_interval_seconds = args.sample_interval_seconds
    if args.gpu_baseline_seconds is not None:
        config.gpu_baseline_seconds = args.gpu_baseline_seconds
    if args.ollama_container_name is not None:
        config.ollama_container_name = args.ollama_container_name
    if args.docker_host is not None:
        config.docker_host = args.docker_host
    if args.ollama_host is not None:
        config.ollama_host = args.ollama_host
    if args.no_unload_models_before_baseline:
        config.unload_models_before_baseline = False
    return config


def run_preflight(config: LocalMetricsConfig) -> dict[str, Any]:
    checks = {
        "ollama_root": _check_ollama_root(config.ollama_host),
        "ollama_loaded_models": _check_ollama_loaded_models(config.ollama_host),
        "docker_version": _check_docker_version(config.docker_host),
        "docker_ollama_container": _check_docker_container(
            config.docker_host,
            config.ollama_container_name,
        ),
        "docker_ollama_stats": _check_docker_stats(
            config.docker_host,
            config.ollama_container_name,
        ),
        "nvidia_smi_gpu": _check_gpu_stats(),
    }
    checks["available"] = all(value.get("ok") for value in checks.values())
    return checks


def run_validation(
    args: argparse.Namespace,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    monitor_factory: Callable[[LocalMetricsConfig], Any] = LocalResourceMonitor,
    monotonic_fn: Callable[[], float] = time.monotonic,
    utcnow_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    config = build_config(args)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = {"skipped": True}
    if not args.skip_preflight:
        preflight = run_preflight(config)

    windows: dict[str, Any] = {
        "pre_pipeline_seconds": args.pre_pipeline_seconds,
        "post_pipeline_seconds": args.post_pipeline_seconds,
    }
    pipeline_return_code = 1
    pipeline_error = None
    measurement = None
    measurement_started_monotonic = None
    summary: dict[str, Any] = {}
    raw_samples: dict[str, Any] = {}

    baseline_started_monotonic = monotonic_fn()
    windows["baseline_started_monotonic"] = _round(baseline_started_monotonic)
    windows["baseline_started_at"] = _iso(utcnow_fn())
    monitor = monitor_factory(config)
    monitor.collect_baseline()
    baseline_finished_monotonic = monotonic_fn()
    windows["baseline_finished_monotonic"] = _round(baseline_finished_monotonic)
    windows["baseline_finished_at"] = _iso(utcnow_fn())

    try:
        windows["external_measurement_started_at"] = _iso(utcnow_fn())
        measurement_started_monotonic = monotonic_fn()
        windows["external_measurement_started_monotonic"] = _round(
            measurement_started_monotonic
        )
        measurement = monitor.start()
        sleep_fn(args.pre_pipeline_seconds)

        windows["pipeline_command_started_at"] = _iso(utcnow_fn())
        pipeline_started_monotonic = monotonic_fn()
        windows["pipeline_command_started_monotonic"] = _round(
            pipeline_started_monotonic
        )
        process = popen_factory(args.command)
        pipeline_return_code = int(process.wait())
        pipeline_finished_monotonic = monotonic_fn()
        windows["pipeline_command_finished_monotonic"] = _round(
            pipeline_finished_monotonic
        )
        windows["pipeline_command_finished_at"] = _iso(utcnow_fn())
        windows["pipeline_runtime_seconds"] = _round(
            pipeline_finished_monotonic - pipeline_started_monotonic
        )

        sleep_fn(args.post_pipeline_seconds)
    except Exception as ex:
        pipeline_error = repr(ex)
    finally:
        if measurement is not None:
            summary = measurement.stop()
            raw_samples = measurement.raw_samples()
        windows["external_measurement_stopped_at"] = _iso(utcnow_fn())
        stopped_monotonic = monotonic_fn()
        windows["external_measurement_stopped_monotonic"] = _round(stopped_monotonic)
        windows["external_measurement_duration_seconds"] = (
            _round(stopped_monotonic - measurement_started_monotonic)
            if measurement_started_monotonic is not None
            else None
        )

    summary = {
        **summary,
        "validation_window": windows,
        "pipeline_command": list(args.command),
        "pipeline_return_code": pipeline_return_code,
        "pipeline_error": pipeline_error,
        "measurement_notes": measurement_notes(),
    }
    raw_payload = {
        **raw_samples,
        "summary": summary,
        "preflight": preflight,
        "validation_window": windows,
        "pipeline_command": list(args.command),
        "pipeline_return_code": pipeline_return_code,
        "pipeline_error": pipeline_error,
    }

    internal_sample_paths = discover_internal_samples(
        args.pipeline_output_dir,
        args.internal_samples,
    )
    event_timeline = build_event_timeline(
        windows,
        raw_payload,
        internal_sample_paths,
        unload_models_before_baseline=config.unload_models_before_baseline,
    )
    summary["event_timeline"] = event_timeline
    raw_payload["summary"] = summary
    raw_payload["event_timeline"] = event_timeline
    _write_yaml(output_dir / "external_local_metrics_samples.yaml", raw_payload)
    _write_yaml(output_dir / "external_local_metrics_summary.yaml", summary)
    _write_yaml(output_dir / "event_timeline.yaml", {"events": event_timeline})

    rows = build_timeseries_rows(
        raw_payload,
        sample_interval_seconds=config.sample_interval_seconds,
        baseline_adjust_gpu=config.baseline_adjust_gpu,
        measurement_start_monotonic=measurement_started_monotonic,
    )
    write_timeseries_csv(output_dir / "external_timeseries.csv", rows)
    generate_plots(
        rows,
        output_dir / "plots",
        args.plots_format,
        raw_payload,
        event_timeline,
    )

    return pipeline_return_code


def build_timeseries_rows(
    raw_payload: dict[str, Any],
    *,
    sample_interval_seconds: float,
    baseline_adjust_gpu: bool,
    measurement_start_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    docker_samples = [
        _docker_sample_from_mapping(sample)
        for sample in raw_payload.get("docker_samples", [])
    ]
    baseline_samples = [
        _gpu_sample_from_mapping(sample)
        for sample in raw_payload.get("baseline_gpu_samples", [])
    ]
    gpu_samples = [
        _gpu_sample_from_mapping(sample)
        for sample in raw_payload.get("gpu_samples", [])
    ]
    start = measurement_start_monotonic or _first_timestamp(
        docker_samples,
        gpu_samples,
    )
    rows.extend(_docker_rows(docker_samples, start))
    rows.extend(
        _gpu_rows(
            gpu_samples,
            baseline_samples if baseline_adjust_gpu else [],
            start,
            sample_interval_seconds,
        )
    )
    return sorted(rows, key=lambda row: (row["relative_seconds"], row["sample_kind"]))


def write_timeseries_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIME_SERIES_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in TIME_SERIES_FIELDS})


def generate_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    plots_format: str,
    raw_payload: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    _line_plot(
        rows,
        output_dir / f"cpu_usage.{plots_format}",
        "CPU Usage",
        [
            ("container_cpu_percent_normalized", "CPU %"),
        ],
        plt,
        events,
    )
    _line_plot(
        rows,
        output_dir / f"memory_ram.{plots_format}",
        "Memory (RAM)",
        [
            ("container_ram_mib", "Process RSS MiB"),
            ("container_ram_cgroup_working_set_mib", "Cgroup working set MiB"),
            ("container_ram_including_cache_mib", "Cgroup including cache MiB"),
        ],
        plt,
        events,
    )
    _line_plot(
        rows,
        output_dir / f"gpu_usage.{plots_format}",
        "GPU Usage",
        [
            ("gpu_utilization_percent_raw", "GPU % raw"),
        ],
        plt,
        events,
    )
    _line_plot(
        rows,
        output_dir / f"gpu_memory_vram.{plots_format}",
        "GPU Memory (VRAM)",
        [
            ("gpu_memory_used_mib_raw", "Memory MiB raw"),
        ],
        plt,
        events,
    )
    _line_plot(
        rows,
        output_dir / f"gpu_power.{plots_format}",
        "GPU Power",
        [
            ("gpu_power_w_raw", "Power W raw"),
        ],
        plt,
        events,
    )
    _polling_plot(raw_payload or {}, output_dir / f"polling_telemetry.{plots_format}", plt)


def discover_internal_samples(
    pipeline_output_dir: Path | None,
    explicit_paths: list[Path],
) -> list[Path]:
    paths = [path.expanduser() for path in explicit_paths]
    if pipeline_output_dir is not None and pipeline_output_dir.exists():
        paths.extend(
            sorted(
                pipeline_output_dir.expanduser().rglob("*_local_metrics_samples.yaml")
            )
        )
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_event_timeline(
    windows: dict[str, Any],
    raw_payload: dict[str, Any],
    internal_sample_paths: list[Path],
    *,
    unload_models_before_baseline: bool,
) -> list[dict[str, Any]]:
    measurement_start = windows.get("external_measurement_started_monotonic")
    if measurement_start is None:
        measurement_start = _first_sample_timestamp(raw_payload)
    events = []
    baseline_label = (
        "Unload models / GPU baseline start"
        if unload_models_before_baseline
        else "GPU baseline start"
    )
    _append_event(
        events,
        "baseline_started",
        baseline_label,
        "baseline",
        windows.get("baseline_started_monotonic"),
        measurement_start,
        wall_time=windows.get("baseline_started_at"),
    )
    _append_event(
        events,
        "baseline_finished",
        "GPU baseline finished",
        "baseline",
        windows.get("baseline_finished_monotonic"),
        measurement_start,
        wall_time=windows.get("baseline_finished_at"),
    )
    _append_event(
        events,
        "external_measurement_started",
        "External measurement start",
        "external",
        windows.get("external_measurement_started_monotonic"),
        measurement_start,
        wall_time=windows.get("external_measurement_started_at"),
    )
    _append_event(
        events,
        "pipeline_started",
        "Pipeline start",
        "pipeline",
        windows.get("pipeline_command_started_monotonic"),
        measurement_start,
        wall_time=windows.get("pipeline_command_started_at"),
    )
    _append_event(
        events,
        "pipeline_finished",
        "Pipeline finish",
        "pipeline",
        windows.get("pipeline_command_finished_monotonic"),
        measurement_start,
        wall_time=windows.get("pipeline_command_finished_at"),
    )
    _append_event(
        events,
        "external_measurement_stopped",
        "External measurement stop",
        "external",
        windows.get("external_measurement_stopped_monotonic"),
        measurement_start,
        wall_time=windows.get("external_measurement_stopped_at"),
    )
    for path in internal_sample_paths:
        events.extend(_configuration_events(path, measurement_start))
    return sorted(events, key=lambda event: event["relative_seconds"])


def _append_event(
    events: list[dict[str, Any]],
    name: str,
    label: str,
    kind: str,
    timestamp_monotonic: Any,
    measurement_start: float | None,
    *,
    wall_time: str | None = None,
    source_path: str | None = None,
) -> None:
    timestamp = _coerce_float(timestamp_monotonic)
    if timestamp is None or measurement_start is None:
        return
    events.append(
        {
            "name": name,
            "label": label,
            "short_label": _short_event_label(label),
            "kind": kind,
            "timestamp_monotonic": _round(timestamp),
            "relative_seconds": _round(timestamp - measurement_start),
            "wall_time": wall_time,
            "source_path": source_path,
        }
    )


def _configuration_events(path: Path, measurement_start: float | None) -> list[dict[str, Any]]:
    if measurement_start is None:
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return []
    timestamps = _runtime_sample_timestamps(payload)
    if not timestamps:
        return []
    label = _configuration_label(path)
    events: list[dict[str, Any]] = []
    baseline_timestamps = _baseline_sample_timestamps(payload)
    if baseline_timestamps:
        _append_event(
            events,
            f"configuration_baseline_started:{label}",
            f"{label} baseline samples start",
            "configuration_baseline",
            min(baseline_timestamps),
            measurement_start,
            source_path=str(path),
        )
        _append_event(
            events,
            f"configuration_baseline_finished:{label}",
            f"{label} baseline samples finish",
            "configuration_baseline",
            max(baseline_timestamps),
            measurement_start,
            source_path=str(path),
        )
    _append_event(
        events,
        f"configuration_started:{label}",
        f"{label} measurement start",
        "configuration",
        min(timestamps),
        measurement_start,
        source_path=str(path),
    )
    _append_event(
        events,
        f"configuration_finished:{label}",
        f"{label} measurement finish",
        "configuration",
        max(timestamps),
        measurement_start,
        source_path=str(path),
    )
    return events


def _runtime_sample_timestamps(payload: dict[str, Any]) -> list[float]:
    timestamps = []
    for key in ["docker_samples", "gpu_samples"]:
        for sample in payload.get(key, []) or []:
            timestamp = _coerce_float(sample.get("timestamp_monotonic"))
            if timestamp is not None:
                timestamps.append(timestamp)
    poll_timestamps = payload.get("poll_timestamps") or {}
    for values in poll_timestamps.values():
        for value in values or []:
            timestamp = _coerce_float(value)
            if timestamp is not None:
                timestamps.append(timestamp)
    return timestamps


def _baseline_sample_timestamps(payload: dict[str, Any]) -> list[float]:
    timestamps = []
    for sample in payload.get("baseline_gpu_samples", []) or []:
        timestamp = _coerce_float(sample.get("timestamp_monotonic"))
        if timestamp is not None:
            timestamps.append(timestamp)
    return timestamps


def _first_sample_timestamp(raw_payload: dict[str, Any]) -> float | None:
    timestamps = _runtime_sample_timestamps(raw_payload)
    timestamps.extend(_baseline_sample_timestamps(raw_payload))
    return min(timestamps) if timestamps else None


def _configuration_label(path: Path) -> str:
    stem = path.stem
    suffix = "_local_metrics_samples"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) == 2 and parts[0] == parts[1]:
        return parts[0]
    midpoint = len(parts) // 2
    if len(parts) % 2 == 0 and parts[:midpoint] == parts[midpoint:]:
        return "_".join(parts[:midpoint])
    return stem


def _short_event_label(label: str) -> str:
    replacements = {
        "Unload models / GPU baseline start": "baseline start",
        "GPU baseline start": "baseline start",
        "GPU baseline finished": "baseline done",
        "External measurement start": "external start",
        "External measurement stop": "external stop",
        "Pipeline start": "pipeline start",
        "Pipeline finish": "pipeline finish",
    }
    if label in replacements:
        return replacements[label]
    if label.endswith(" baseline samples start"):
        return label[: -len(" baseline samples start")].split("_")[-1] + " baseline"
    if label.endswith(" baseline samples finish"):
        return label[: -len(" baseline samples finish")].split("_")[-1] + " baseline done"
    if label.endswith(" measurement start"):
        return label[: -len(" measurement start")].split("_")[-1] + " start"
    if label.endswith(" measurement finish"):
        return label[: -len(" measurement finish")].split("_")[-1] + " finish"
    if label.endswith(" start"):
        return label[: -len(" start")].split("_")[-1] + " start"
    if label.endswith(" finish"):
        return label[: -len(" finish")].split("_")[-1] + " finish"
    return label


def measurement_notes() -> list[str]:
    return [
        "Docker CPU percentages require at least two container stats samples.",
        (
            "The requested sampling interval may not be achievable because Docker "
            "stats and nvidia-smi calls have nontrivial overhead."
        ),
        (
            "RAM uses the Ollama container process RSS from Docker top when "
            "available. Cgroup working-set and cgroup usage-including-cache "
            "fields are retained because memory-mapped model files can otherwise "
            "make Docker memory accounting misleading."
        ),
        (
            "Ollama response runtime/token metrics cannot be independently sampled "
            "externally; compare them from pipeline result/debug files instead."
        ),
    ]


def _docker_rows(
    samples: list[DockerContainerStatsSample],
    start: float,
) -> list[dict[str, Any]]:
    rows = []
    previous = None
    for sample in samples:
        cpu_percent = calculate_cpu_percent(previous, sample)
        normalized = None
        if cpu_percent is not None and sample.online_cpus:
            normalized = round(cpu_percent / sample.online_cpus, 6)
        rows.append(
            _base_row(
                "docker",
                sample.timestamp_monotonic,
                start,
                container_cpu_percent=cpu_percent,
                container_cpu_percent_normalized=normalized,
                container_ram_bytes=sample.memory_usage_bytes,
                container_process_rss_bytes=sample.process_rss_bytes,
                container_process_vsz_bytes=sample.process_vsz_bytes,
                container_process_count=sample.process_count,
                container_ram_cgroup_working_set_bytes=(
                    sample.memory_cgroup_working_set_bytes
                ),
                container_ram_including_cache_bytes=(
                    sample.memory_usage_including_cache_bytes
                ),
                container_ram_file_bytes=sample.memory_file_bytes,
                container_ram_active_file_bytes=sample.memory_active_file_bytes,
                container_ram_inactive_file_bytes=sample.memory_inactive_file_bytes,
                container_ram_anon_bytes=sample.memory_anon_bytes,
                container_ram_limit_bytes=sample.memory_limit_bytes,
                container_pids=sample.pids_current,
            )
        )
        previous = sample
    return rows


def _gpu_rows(
    samples: list[GpuStatsSample],
    baseline_samples: list[GpuStatsSample],
    start: float,
    sample_interval_seconds: float,
) -> list[dict[str, Any]]:
    baseline_memory = _avg(
        sample.memory_used_mib
        for sample in baseline_samples
        if sample.memory_used_mib is not None
    )
    baseline_util = _avg(
        sample.utilization_gpu_percent
        for sample in baseline_samples
        if sample.utilization_gpu_percent is not None
    )
    baseline_power = _avg(
        sample.power_draw_w
        for sample in baseline_samples
        if sample.power_draw_w is not None
    )
    sorted_samples = sorted(samples, key=lambda sample: sample.timestamp_monotonic)
    intervals = _intervals(sorted_samples, sample_interval_seconds)
    cumulative_energy = 0.0
    rows = []
    for sample, interval in zip(sorted_samples, intervals):
        adjusted_memory = _adjust(sample.memory_used_mib, baseline_memory)
        adjusted_util = _adjust(sample.utilization_gpu_percent, baseline_util)
        adjusted_power = _adjust(sample.power_draw_w, baseline_power)
        if adjusted_power is not None:
            cumulative_energy += adjusted_power * interval / 3600.0
        rows.append(
            _base_row(
                "gpu",
                sample.timestamp_monotonic,
                start,
                gpu_index=sample.gpu_index,
                gpu_name=sample.name,
                gpu_memory_used_mib_raw=sample.memory_used_mib,
                gpu_memory_used_mib_adjusted=adjusted_memory,
                gpu_utilization_percent_raw=sample.utilization_gpu_percent,
                gpu_utilization_percent_adjusted=adjusted_util,
                gpu_memory_utilization_percent=sample.utilization_memory_percent,
                gpu_power_w_raw=sample.power_draw_w,
                gpu_power_w_adjusted=adjusted_power,
                gpu_energy_wh_adjusted_cumulative=round(cumulative_energy, 9),
            )
        )
    return rows


def _line_plot(
    rows,
    path: Path,
    title: str,
    series: list[tuple[str, str]],
    plt,
    events: list[dict[str, Any]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = False
    for key, label in series:
        points = []
        for row in rows:
            value = _plot_value(row, key)
            if value is not None:
                points.append((row["relative_seconds"], value))
        if not points:
            continue
        x, y = zip(*points)
        ax.plot(x, y, label=label)
        plotted = True
    _annotate_events(ax, events or [])
    ax.set_title(title)
    ax.set_xlabel("Seconds since external measurement start")
    ax.grid(True, alpha=0.3)
    if plotted:
        _place_compact_below_plot_legend(ax)
    else:
        ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _plot_value(row: dict[str, Any], key: str) -> float | None:
    mib_keys = {
        "container_ram_mib": "container_ram_bytes",
        "container_ram_cgroup_working_set_mib": (
            "container_ram_cgroup_working_set_bytes"
        ),
        "container_ram_including_cache_mib": "container_ram_including_cache_bytes",
    }
    if key in mib_keys:
        value = row.get(mib_keys[key])
        if value is None:
            return None
        return float(value) / (1024.0 * 1024.0)
    value = row.get(key)
    return float(value) if value is not None else None


def _polling_plot(raw_payload: dict[str, Any], path: Path, plt) -> None:
    durations = raw_payload.get("poll_durations_seconds") or {}
    fig, ax = plt.subplots(figsize=(13, 5))
    plotted = False
    for key, values in durations.items():
        numeric = [value for value in values if value is not None]
        if not numeric:
            continue
        ax.plot(range(len(numeric)), numeric, label=key)
        plotted = True
    ax.set_title("Polling Durations")
    ax.set_xlabel("Poll index")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)
    if plotted:
        _place_compact_below_plot_legend(ax)
    else:
        ax.text(0.5, 0.5, "No polling telemetry", transform=ax.transAxes, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _place_compact_below_plot_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = min(3, max(1, len(labels)))
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=ncol,
        fontsize=9,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
    )


def _annotate_events(ax, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    seen_labels = set()
    for index, event in enumerate(events):
        x = event.get("relative_seconds")
        if x is None:
            continue
        style = EVENT_STYLES.get(event.get("kind"), EVENT_STYLES["external"])
        legend_label = event.get("label") if event.get("name") not in seen_labels else None
        ax.axvline(
            x,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.1,
            alpha=0.75,
            label=legend_label,
        )
        seen_labels.add(event.get("name"))
        if index % 2 == 0:
            ax.text(
                x,
                0.98,
                event.get("short_label") or event.get("label") or event.get("name"),
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color=style["color"],
                alpha=0.9,
            )
    x_values = [event.get("relative_seconds") for event in events]
    x_values = [value for value in x_values if value is not None]
    if x_values:
        current_left, current_right = ax.get_xlim()
        ax.set_xlim(min(current_left, min(x_values)), max(current_right, max(x_values)))


def _check_ollama_root(ollama_host: str) -> dict[str, Any]:
    try:
        body = _urlopen_text(urllib.parse.urljoin(ollama_host.rstrip("/") + "/", "/"))
        return {"ok": True, "message": body.strip()[:200]}
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


def _check_ollama_loaded_models(ollama_host: str) -> dict[str, Any]:
    try:
        body = _urlopen_text(
            urllib.parse.urljoin(ollama_host.rstrip("/") + "/", "/api/ps")
        )
        data = json.loads(body) if body else {}
        return {"ok": True, "model_count": len(data.get("models") or [])}
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


def _check_docker_version(docker_host: str) -> dict[str, Any]:
    try:
        version = DockerProxyClient(docker_host).version()
        return {
            "ok": True,
            "version": version.get("Version"),
            "api_version": version.get("ApiVersion"),
        }
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


def _check_docker_container(docker_host: str, container_name: str) -> dict[str, Any]:
    try:
        inspect = DockerProxyClient(docker_host).inspect_container(container_name)
        return {
            "ok": True,
            "id": inspect.get("Id"),
            "name": inspect.get("Name"),
            "state": (inspect.get("State") or {}).get("Status"),
        }
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


def _check_docker_stats(docker_host: str, container_name: str) -> dict[str, Any]:
    try:
        sample = DockerProxyClient(docker_host).container_stats(container_name)
        return {"ok": True, "sample": asdict(sample)}
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


def _check_gpu_stats() -> dict[str, Any]:
    try:
        samples = sample_gpu_stats()
        return {"ok": True, "gpu_count": len(samples)}
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


def _urlopen_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _docker_sample_from_mapping(data: dict[str, Any]) -> DockerContainerStatsSample:
    return DockerContainerStatsSample(**data)


def _gpu_sample_from_mapping(data: dict[str, Any]) -> GpuStatsSample:
    return GpuStatsSample(**data)


def _base_row(sample_kind: str, timestamp: float, start: float, **values) -> dict[str, Any]:
    row = {field: None for field in TIME_SERIES_FIELDS}
    row.update(
        {
            "source": "external",
            "sample_kind": sample_kind,
            "timestamp_monotonic": timestamp,
            "relative_seconds": round(timestamp - start, 6),
        }
    )
    row.update(values)
    return row


def _first_timestamp(*sample_groups) -> float:
    timestamps = [
        sample.timestamp_monotonic
        for group in sample_groups
        for sample in group
        if sample.timestamp_monotonic is not None
    ]
    return min(timestamps) if timestamps else 0.0


def _intervals(samples: list[GpuStatsSample], fallback: float) -> list[float]:
    if not samples:
        return []
    timestamps = [sample.timestamp_monotonic for sample in samples]
    intervals = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current >= previous
    ]
    fallback_interval = _avg(intervals) or fallback
    return intervals + [fallback_interval]


def _avg(values) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _adjust(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return max(0.0, value - baseline)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def main(argv: list[str] | None = None) -> int:
    return run_validation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
