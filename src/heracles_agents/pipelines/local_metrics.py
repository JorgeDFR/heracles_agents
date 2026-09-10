"""Pipeline helpers for optional local resource metrics."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from heracles_agents.agent_functions import generate_prompt_for_agent
from heracles_agents.inference_parameters import get_reasoning_settings
from heracles_agents.llm_interface import LocalResourceMetrics, generate_tools_for_agent
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
    """Preload and warm each Ollama model after baseline, before sampling."""

    config = monitor.config
    if not config.enabled or not config.warmup_enabled:
        monitor.warmup = {"enabled": False, "requests": []}
        return

    phases_by_model = {}
    for phase in getattr(exp, "phases", {}).values():
        if getattr(getattr(phase, "client", None), "client_type", None) != "ollama":
            continue
        phases_by_model.setdefault(str(phase.model_info.model), phase)
    records = []
    for model, phase in phases_by_model.items():
        # The preload request is intentionally separate from the configured
        # warmup requests. When models are unloaded for the baseline, that
        # first request measures the cold start; at least one subsequent
        # request then exercises the already-resident model before questions
        # are timed.
        request_count = config.warmup_requests + 1
        for attempt in range(1, request_count + 1):
            kind = (
                "cold_start"
                if attempt == 1 and config.unload_models_before_baseline
                else "preload" if attempt == 1 else "warmup"
            )
            logger.info(
                "Preparing Ollama model: %s [%s] (%s/%s)",
                model,
                kind,
                attempt,
                request_count,
            )
            payload, profile = _ollama_warmup_payload(
                phase,
                config,
                use_benchmark_prompt=kind == "warmup",
            )
            request = urllib.request.Request(
                config.ollama_host.rstrip("/") + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
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
                        "kind": kind,
                        "profile": profile,
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
                        "kind": kind,
                        "profile": profile,
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
                for model in phases_by_model
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


def _ollama_warmup_payload(
    phase,
    config: LocalMetricsConfig,
    *,
    use_benchmark_prompt: bool,
) -> tuple[dict[str, Any], str]:
    """Build a short warmup that primes the real prompt/tool request shape."""

    messages = [{"role": "user", "content": config.warmup_prompt}]
    tools = []
    profile = "model_load"
    if use_benchmark_prompt:
        try:
            prompt = copy.deepcopy(phase.agent_info.prompt_settings.base_prompt)
            prompt.novel_instruction = config.warmup_prompt
            messages = generate_prompt_for_agent(prompt, phase)
            tools = generate_tools_for_agent(phase.agent_info)
            profile = "benchmark_prompt"
        except Exception as ex:
            logger.warning(
                "Unable to render benchmark-shaped Ollama warmup; using the short "
                "warmup prompt instead: %s",
                ex,
            )
            profile = "short_prompt_fallback"

    options = {"temperature": 0, "num_predict": 8}
    seed = getattr(phase.model_info, "seed", None)
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": phase.model_info.model,
        "messages": messages,
        "stream": False,
        "keep_alive": config.warmup_keep_alive,
        "options": options,
    }
    if tools:
        payload["tools"] = tools

    reasoning_mode, reasoning_effort = get_reasoning_settings(phase.model_info)
    if reasoning_mode == "enabled":
        payload["think"] = reasoning_effort or True
    elif reasoning_mode == "disabled":
        payload["think"] = False
    return payload, profile


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
