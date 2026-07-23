"""Pipeline helpers for optional local resource metrics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from heracles_agents.llm_interface import LocalResourceMetrics
from heracles_agents.local_metrics.monitor import (
    LocalMetricsConfig,
    LocalResourceMeasurement,
    LocalResourceMonitor,
    config_from_mapping,
)
from heracles_agents.local_metrics.ollama_runtime import summarize_ollama_metrics


def prepare_local_resource_monitor(exp) -> LocalResourceMonitor:
    config = config_from_mapping(getattr(exp, "local_metrics", None))
    if config.target_provider != "ollama" or not _uses_provider(exp, "ollama"):
        config.enabled = False
    monitor = LocalResourceMonitor(config)
    monitor.collect_baseline()
    return monitor


def start_local_resource_measurement(
    monitor: LocalResourceMonitor,
) -> LocalResourceMeasurement:
    return monitor.start()


def make_local_resource_metrics(
    contexts: list[Any],
    measurement: LocalResourceMeasurement,
) -> LocalResourceMetrics | None:
    if not measurement.config.enabled:
        return None
    ollama_metrics = [
        metric
        for context in contexts
        for metric in getattr(context, "local_llm_runtime_metrics", [])
    ]
    summary = measurement.stop()
    summary["ollama"] = summarize_ollama_metrics(ollama_metrics)
    _write_debug_samples(measurement, ollama_metrics, summary)
    return LocalResourceMetrics(**summary)


def _uses_provider(exp, provider_name: str) -> bool:
    for phase in getattr(exp, "phases", {}).values():
        provider = getattr(getattr(phase, "client", None), "client_type", None)
        if provider == provider_name:
            return True
    return False


def _write_debug_samples(
    measurement: LocalResourceMeasurement,
    ollama_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if not measurement.config.record_samples:
        return
    output_path = measurement.config.samples_output_path
    if not output_path:
        output_path = "output/local_metrics_samples.yaml"
    path = Path(os.path.expandvars(output_path)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = measurement.raw_samples()
    payload["ollama_runtime_metrics"] = ollama_metrics
    payload["summary"] = summary
    with path.open("w", encoding="utf-8") as fo:
        yaml.safe_dump(payload, fo, sort_keys=False)
