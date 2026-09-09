"""Pipeline helpers for optional local resource metrics."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
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

logger = logging.getLogger(__name__)


def prepare_local_resource_monitor(exp) -> LocalResourceMonitor:
    config = config_from_mapping(getattr(exp, "local_metrics", None))
    if config.target_provider != "ollama" or not _uses_provider(exp, "ollama"):
        config.enabled = False
    monitor = LocalResourceMonitor(config)
    monitor.collect_baseline()
    _warm_up_ollama(exp, monitor)
    return monitor


def _warm_up_ollama(exp, monitor: LocalResourceMonitor) -> None:
    """Load each configuration's Ollama model after baseline, before sampling."""

    config = monitor.config
    if not config.enabled or not config.warmup_enabled:
        monitor.warmup = {"enabled": False, "requests": []}
        return

    models = list(
        dict.fromkeys(
            str(phase.model_info.model)
            for phase in getattr(exp, "phases", {}).values()
            if getattr(getattr(phase, "client", None), "client_type", None) == "ollama"
        )
    )
    records = []
    for model in models:
        for attempt in range(1, config.warmup_requests + 1):
            logger.info(
                "Warming Ollama model: %s (%s/%s)",
                model,
                attempt,
                config.warmup_requests,
            )
            payload = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": config.warmup_prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 8},
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                config.ollama_host.rstrip("/") + "/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(
                    request, timeout=config.warmup_timeout_seconds
                ) as response:
                    response.read()
                records.append(
                    {
                        "model": model,
                        "request": attempt,
                        "succeeded": True,
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                    }
                )
            except Exception as ex:
                warning = f"Ollama warmup failed for {model}: {type(ex).__name__}: {ex}"
                monitor.warnings.append(warning)
                records.append(
                    {
                        "model": model,
                        "request": attempt,
                        "succeeded": False,
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        "error": warning,
                    }
                )
    monitor.warmup = {"enabled": True, "requests": records}


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
