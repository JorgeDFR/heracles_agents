"""Background local resource monitor for Ollama benchmark runs."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from heracles_agents.local_metrics.docker_proxy import (
    DockerContainerStatsSample,
    DockerProxyClient,
    summarize_docker_samples,
)
from heracles_agents.local_metrics.gpu_stats import (
    GpuStatsSample,
    sample_gpu_stats,
    summarize_gpu_samples,
)
from heracles_agents.local_metrics.ollama_runtime import (
    summarize_ollama_metrics,
    unload_loaded_models,
)


@dataclass
class LocalMetricsConfig:
    enabled: bool = False
    target_provider: str = "ollama"
    ollama_container_name: str = "ollama"
    docker_host: str = "tcp://docker-socket-proxy:2375"
    ollama_host: str = "http://ollama:11434"
    sample_interval_seconds: float = 0.025
    gpu_baseline_seconds: float = 5.0
    unload_models_before_baseline: bool = True
    baseline_adjust_gpu: bool = True
    record_samples: bool = False
    samples_output_path: str | None = None
    warmup_enabled: bool = True
    warmup_requests: int = 1
    warmup_prompt: str = "Reply with OK."
    warmup_timeout_seconds: float = 7200.0


@dataclass
class LocalResourceMeasurement:
    config: LocalMetricsConfig
    baseline_gpu_samples: list[GpuStatsSample]
    warmup: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    docker_samples: list[DockerContainerStatsSample] = field(default_factory=list)
    gpu_samples: list[GpuStatsSample] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _threads: list[threading.Thread] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _started_at: float | None = None
    _stopped_at: float | None = None
    _docker_poll_timestamps: list[float] = field(default_factory=list)
    _docker_poll_durations: list[float] = field(default_factory=list)
    _gpu_poll_timestamps: list[float] = field(default_factory=list)
    _gpu_poll_durations: list[float] = field(default_factory=list)

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._started_at = time.monotonic()
        if self.config.docker_host:
            self._threads.append(
                threading.Thread(target=self._docker_sample_loop, daemon=True)
            )
        else:
            self._append_warning("Docker host is not configured.")
        self._threads.append(threading.Thread(target=self._gpu_sample_loop, daemon=True))
        for thread in self._threads:
            thread.start()

    def stop(self) -> dict[str, Any]:
        if self._threads:
            self._stop_event.set()
            for thread in self._threads:
                thread.join(timeout=max(2.0, self.config.sample_interval_seconds * 4))
        self._stopped_at = time.monotonic()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            docker_samples = list(self.docker_samples)
            gpu_samples = list(self.gpu_samples)
            telemetry = self._telemetry_summary_locked()
            warnings = list(self.warnings)
        cpu, ram = summarize_docker_samples(docker_samples)
        gpu = summarize_gpu_samples(
            gpu_samples,
            self.baseline_gpu_samples if self.config.baseline_adjust_gpu else [],
            sample_interval_seconds=self.config.sample_interval_seconds,
        )
        warnings.extend(self._sampling_warnings(telemetry))
        return {
            "measurement_scope": "ollama_container",
            "ollama_container_name": self.config.ollama_container_name,
            "sample_interval_seconds": self.config.sample_interval_seconds,
            "baseline_seconds": self.config.gpu_baseline_seconds,
            "baseline_adjusted": self.config.baseline_adjust_gpu,
            "warmup": dict(self.warmup),
            "telemetry": telemetry,
            "cpu": cpu,
            "ram": ram,
            "gpu": gpu,
            "warnings": list(dict.fromkeys(warnings)),
        }

    def raw_samples(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": {
                    "target_provider": self.config.target_provider,
                    "ollama_container_name": self.config.ollama_container_name,
                    "docker_host": self.config.docker_host,
                    "ollama_host": self.config.ollama_host,
                    "sample_interval_seconds": self.config.sample_interval_seconds,
                    "gpu_baseline_seconds": self.config.gpu_baseline_seconds,
                    "baseline_adjust_gpu": self.config.baseline_adjust_gpu,
                    "warmup_enabled": self.config.warmup_enabled,
                    "warmup_requests": self.config.warmup_requests,
                },
                "telemetry": self._telemetry_summary_locked(),
                "poll_timestamps": {
                    "docker_monotonic_seconds": list(self._docker_poll_timestamps),
                    "gpu_monotonic_seconds": list(self._gpu_poll_timestamps),
                },
                "poll_durations_seconds": {
                    "docker": list(self._docker_poll_durations),
                    "gpu": list(self._gpu_poll_durations),
                },
                "baseline_gpu_samples": [
                    asdict(sample) for sample in self.baseline_gpu_samples
                ],
                "docker_samples": [asdict(sample) for sample in self.docker_samples],
                "gpu_samples": [asdict(sample) for sample in self.gpu_samples],
                "warnings": list(dict.fromkeys(self.warnings)),
            }

    def _docker_sample_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                docker_sample = DockerProxyClient(
                    self.config.docker_host
                ).container_stats(self.config.ollama_container_name)
            except Exception as ex:
                self._append_warning(f"Docker proxy metrics unavailable: {ex}")
                docker_sample = None
            duration = time.monotonic() - started
            with self._lock:
                self._docker_poll_timestamps.append(started)
                self._docker_poll_durations.append(duration)
                if docker_sample is not None:
                    self.docker_samples.append(docker_sample)
            self._wait_for_next_poll(started)

    def _gpu_sample_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                gpu_samples = sample_gpu_stats()
            except Exception as ex:
                self._append_warning(f"nvidia-smi GPU metrics unavailable: {ex}")
                gpu_samples = []
            duration = time.monotonic() - started
            with self._lock:
                self._gpu_poll_timestamps.append(started)
                self._gpu_poll_durations.append(duration)
                self.gpu_samples.extend(gpu_samples)
            self._wait_for_next_poll(started)

    def _sample_once(self) -> None:
        """Collect one synchronous sample for focused unit tests."""
        started = time.monotonic()
        try:
            gpu_samples = sample_gpu_stats()
        except Exception as ex:
            self._append_warning(f"nvidia-smi GPU metrics unavailable: {ex}")
            gpu_samples = []

        docker_sample = None
        if self.config.docker_host:
            try:
                docker_sample = DockerProxyClient(
                    self.config.docker_host
                ).container_stats(self.config.ollama_container_name)
            except Exception as ex:
                self._append_warning(f"Docker proxy metrics unavailable: {ex}")
        else:
            self._append_warning("Docker host is not configured.")

        with self._lock:
            if docker_sample is not None:
                self.docker_samples.append(docker_sample)
                self._docker_poll_timestamps.append(started)
                self._docker_poll_durations.append(time.monotonic() - started)
            self._gpu_poll_timestamps.append(started)
            self.gpu_samples.extend(gpu_samples)

    def _wait_for_next_poll(self, started: float) -> None:
        elapsed = time.monotonic() - started
        delay = max(0.0, self.config.sample_interval_seconds - elapsed)
        self._stop_event.wait(delay)

    def _append_warning(self, warning: str) -> None:
        with self._lock:
            self.warnings.append(warning)

    def _telemetry_summary_locked(self) -> dict[str, Any]:
        return {
            "requested_sample_interval_seconds": self.config.sample_interval_seconds,
            "measurement_duration_seconds": _round_or_none(
                (self._stopped_at or time.monotonic()) - self._started_at
                if self._started_at is not None
                else None
            ),
            "docker_poll_count": len(self._docker_poll_timestamps),
            "docker_effective_interval_seconds_avg": _avg_intervals(
                self._docker_poll_timestamps
            ),
            "docker_poll_duration_seconds_avg": _avg(self._docker_poll_durations),
            "gpu_poll_count": len(self._gpu_poll_timestamps),
            "gpu_effective_interval_seconds_avg": _avg_intervals(
                self._gpu_poll_timestamps
            ),
            "gpu_poll_duration_seconds_avg": _avg(self._gpu_poll_durations),
        }

    def _sampling_warnings(self, telemetry: dict[str, Any]) -> list[str]:
        warnings = []
        requested = self.config.sample_interval_seconds
        for label, key in [
            ("Docker", "docker_effective_interval_seconds_avg"),
            ("GPU", "gpu_effective_interval_seconds_avg"),
        ]:
            effective = telemetry.get(key)
            if (
                requested > 0
                and effective is not None
                and effective > requested * 2
            ):
                warnings.append(
                    f"Requested local metrics sample interval {requested:.4f}s "
                    f"was not achievable for {label}; effective average was "
                    f"{effective:.4f}s."
                )
        if telemetry.get("docker_poll_count", 0) < 2:
            warnings.append(
                "Docker CPU percentage needs at least two container stats samples; "
                "short questions may report null CPU metrics."
            )
        return warnings


class LocalResourceMonitor:
    def __init__(self, config: LocalMetricsConfig):
        self.config = config
        self.baseline_gpu_samples: list[GpuStatsSample] = []
        self.warnings: list[str] = []
        self._baseline_collected = False
        self.warmup: dict[str, Any] = {}

    def collect_baseline(self) -> None:
        if not self.config.enabled or self._baseline_collected:
            return
        if self.config.unload_models_before_baseline:
            try:
                unloaded = unload_loaded_models(self.config.ollama_host)
                if unloaded:
                    self.warnings.append(
                        "Unloaded Ollama models before baseline: "
                        + ", ".join(unloaded)
                    )
            except Exception as ex:
                self.warnings.append(f"Unable to unload Ollama models: {ex}")

        deadline = time.monotonic() + max(0.0, self.config.gpu_baseline_seconds)
        while time.monotonic() < deadline:
            started = time.monotonic()
            try:
                self.baseline_gpu_samples.extend(sample_gpu_stats())
            except Exception as ex:
                self.warnings.append(f"Unable to collect GPU baseline: {ex}")
                break
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, self.config.sample_interval_seconds - elapsed))
        self._baseline_collected = True

    def start(self) -> LocalResourceMeasurement:
        measurement = LocalResourceMeasurement(
            config=self.config,
            baseline_gpu_samples=list(self.baseline_gpu_samples),
            warmup=dict(self.warmup),
            warnings=list(self.warnings),
        )
        measurement.start()
        return measurement

    def stop(self) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {"warnings": list(self.warnings)}


def config_from_mapping(data: dict[str, Any] | None) -> LocalMetricsConfig:
    data = data or {}
    enabled = _as_bool(data.get("enabled"), _env_bool("LOCAL_METRICS_ENABLED", False))
    return LocalMetricsConfig(
        enabled=enabled,
        target_provider=str(
            data.get("target_provider")
            or "ollama"
        ),
        ollama_container_name=str(
            data.get("ollama_container_name")
            or "ollama"
        ),
        docker_host=str(
            data.get("docker_host")
            or os.environ.get("DOCKER_HOST")
            or "tcp://docker-socket-proxy:2375"
        ),
        ollama_host=str(
            data.get("ollama_host")
            or os.environ.get("OLLAMA_HOST")
            or "http://ollama:11434"
        ),
        sample_interval_seconds=float(
            data.get("sample_interval_seconds", 0.025)
        ),
        gpu_baseline_seconds=float(
            data.get("gpu_baseline_seconds", 5.0)
        ),
        unload_models_before_baseline=_as_bool(
            data.get("unload_models_before_baseline"),
            True,
        ),
        baseline_adjust_gpu=_as_bool(
            data.get("baseline_adjust_gpu"),
            True
        ),
        record_samples=_as_bool(
            data.get("record_samples"),
            False,
        ),
        samples_output_path=(
            str(data.get("samples_output_path"))
            if data.get("samples_output_path")
            else None
        ),
        warmup_enabled=_as_bool(data.get("warmup_enabled"), True),
        warmup_requests=max(1, int(data.get("warmup_requests", 1))),
        warmup_prompt=str(data.get("warmup_prompt") or "Reply with OK."),
        warmup_timeout_seconds=float(data.get("warmup_timeout_seconds", 7200)),
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _env_bool(name: str, default: bool) -> bool:
    return _as_bool(os.environ.get(name), default)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _avg_intervals(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    intervals = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current >= previous
    ]
    return _avg(intervals)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
