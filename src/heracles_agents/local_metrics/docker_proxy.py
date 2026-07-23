"""Read-only Docker Engine metrics through a Docker socket proxy."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class DockerContainerStatsSample:
    timestamp_monotonic: float
    cpu_total_usage: int | None
    system_cpu_usage: int | None
    online_cpus: int | None
    memory_usage_bytes: int | None
    memory_limit_bytes: int | None
    pids_current: int | None


class DockerProxyClient:
    def __init__(self, docker_host: str, *, timeout_seconds: float = 5.0):
        self.base_url = docker_host_to_http_base_url(docker_host)
        self.timeout_seconds = timeout_seconds

    def version(self) -> dict[str, Any]:
        return self._get_json("/version")

    def list_containers(self) -> list[dict[str, Any]]:
        data = self._get_json("/containers/json")
        return data if isinstance(data, list) else []

    def inspect_container(self, container_name: str) -> dict[str, Any]:
        return self._get_json(f"/containers/{_quote_container(container_name)}/json")

    def container_stats(self, container_name: str) -> DockerContainerStatsSample:
        data = self._get_json(
            f"/containers/{_quote_container(container_name)}/stats?"
            "stream=false&one-shot=true"
        )
        return docker_stats_sample_from_dict(data)

    def _get_json(self, path: str) -> Any:
        request = urllib.request.Request(
            urllib.parse.urljoin(self.base_url + "/", path.lstrip("/")),
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def docker_host_to_http_base_url(docker_host: str) -> str:
    if not docker_host:
        raise ValueError("Docker host is not configured.")
    if docker_host.startswith("tcp://"):
        return "http://" + docker_host[len("tcp://") :].rstrip("/")
    if docker_host.startswith("http://"):
        return docker_host.rstrip("/")
    raise ValueError(
        f"Unsupported Docker host scheme for local metrics: {docker_host!r}."
    )


def docker_stats_sample_from_dict(data: dict[str, Any]) -> DockerContainerStatsSample:
    cpu_stats = data.get("cpu_stats") or {}
    cpu_usage = cpu_stats.get("cpu_usage") or {}
    memory_stats = data.get("memory_stats") or {}
    pids_stats = data.get("pids_stats") or {}
    return DockerContainerStatsSample(
        timestamp_monotonic=time.monotonic(),
        cpu_total_usage=_coerce_int(cpu_usage.get("total_usage")),
        system_cpu_usage=_coerce_int(cpu_stats.get("system_cpu_usage")),
        online_cpus=_coerce_int(cpu_stats.get("online_cpus")),
        memory_usage_bytes=_coerce_int(memory_stats.get("usage")),
        memory_limit_bytes=_coerce_int(memory_stats.get("limit")),
        pids_current=_coerce_int(pids_stats.get("current")),
    )


def calculate_cpu_percent(
    previous: DockerContainerStatsSample | None,
    current: DockerContainerStatsSample | None,
) -> float | None:
    if previous is None or current is None:
        return None
    if (
        previous.cpu_total_usage is None
        or current.cpu_total_usage is None
        or previous.system_cpu_usage is None
        or current.system_cpu_usage is None
    ):
        return None
    cpu_delta = current.cpu_total_usage - previous.cpu_total_usage
    system_delta = current.system_cpu_usage - previous.system_cpu_usage
    online_cpus = current.online_cpus or previous.online_cpus or 1
    if cpu_delta <= 0 or system_delta <= 0:
        return None
    return round((cpu_delta / system_delta) * online_cpus * 100.0, 6)


def summarize_docker_samples(
    samples: list[DockerContainerStatsSample],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu_values = []
    cpu_normalized_values = []
    for previous, current in zip(samples, samples[1:]):
        value = calculate_cpu_percent(previous, current)
        if value is None:
            continue
        cpu_values.append(value)
        online_cpus = current.online_cpus or previous.online_cpus
        if online_cpus:
            cpu_normalized_values.append(round(value / online_cpus, 6))
    ram_values = [
        sample.memory_usage_bytes
        for sample in samples
        if sample.memory_usage_bytes is not None
    ]
    limits = [
        sample.memory_limit_bytes
        for sample in samples
        if sample.memory_limit_bytes is not None
    ]
    pids = [sample.pids_current for sample in samples if sample.pids_current is not None]
    cpu = {
        "container_cpu_percent_avg": _avg(cpu_values),
        "container_cpu_percent_peak": max(cpu_values) if cpu_values else None,
        "container_cpu_percent_normalized_avg": _avg(cpu_normalized_values),
        "container_cpu_percent_normalized_peak": max(cpu_normalized_values)
        if cpu_normalized_values
        else None,
        "container_online_cpus": _last_online_cpus(samples),
        "sample_count": len(samples),
    }
    ram = {
        "container_ram_bytes_avg": int(round(_avg(ram_values)))
        if ram_values
        else None,
        "container_ram_bytes_peak": max(ram_values) if ram_values else None,
        "container_ram_limit_bytes": limits[-1] if limits else None,
        "container_pids_peak": max(pids) if pids else None,
        "sample_count": len(samples),
    }
    return cpu, ram


def _last_online_cpus(samples: list[DockerContainerStatsSample]) -> int | None:
    for sample in reversed(samples):
        if sample.online_cpus is not None:
            return sample.online_cpus
    return None


def _quote_container(container_name: str) -> str:
    return urllib.parse.quote(container_name.lstrip("/"), safe="")


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
