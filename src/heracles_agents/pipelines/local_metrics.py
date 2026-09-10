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
from heracles_agents.local_metrics.ollama_runtime import (
    extract_ollama_response_metrics,
    list_loaded_models,
)

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
                    "keep_alive": config.warmup_keep_alive,
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
                    response_data = json.loads(response.read().decode("utf-8"))
                records.append(
                    {
                        "model": model,
                        "request": attempt,
                        "kind": "cold_start"
                        if attempt == 1 and config.unload_models_before_baseline
                        else "warmup",
                        "succeeded": True,
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        **extract_ollama_response_metrics(response_data),
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
    resident_models = []
    resident_verified = None
    if config.warmup_verify_resident:
        try:
            resident_models = [
                str(item.get("name") or item.get("model"))
                for item in list_loaded_models(config.ollama_host)
                if item.get("name") or item.get("model")
            ]
            resident_verified = all(
                any(
                    loaded == model
                    or loaded.removesuffix(":latest") == model.removesuffix(":latest")
                    for loaded in resident_models
                )
                for model in models
            )
            if not resident_verified:
                monitor.warnings.append(
                    "Ollama warmup completed, but not every model was resident."
                )
        except Exception as ex:
            monitor.warnings.append(
                f"Unable to verify Ollama model residency after warmup: {ex}"
            )
    monitor.warmup = {
        "enabled": True,
        "keep_alive": config.warmup_keep_alive,
        "resident_verified": resident_verified,
        "resident_models": resident_models,
        "requests": records,
    }


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
    summary = measurement.stop()
    ollama_metrics = [
        metric
        for context in contexts
        for metric in getattr(context, "local_llm_runtime_metrics", [])
    ]
    _write_debug_samples(measurement, ollama_metrics, summary)
    return LocalResourceMetrics(**summary)


def make_question_local_resource_metrics(
    measurement: LocalResourceMeasurement,
    started_at: float,
    stopped_at: float,
) -> LocalResourceMetrics | None:
    """Return CPU/RAM/GPU telemetry scoped to one question."""

    if not measurement.config.enabled:
        return None
    summary = measurement.summary_between(started_at, stopped_at)
    telemetry = summary.get("telemetry") or {}
    if not telemetry.get("docker_sample_count") and not telemetry.get(
        "gpu_sample_count"
    ):
        return None
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
