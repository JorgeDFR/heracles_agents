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
    memory_cgroup_working_set_bytes: int | None = None
    memory_usage_including_cache_bytes: int | None = None
    memory_file_bytes: int | None = None
    memory_active_file_bytes: int | None = None
    memory_inactive_file_bytes: int | None = None
    memory_anon_bytes: int | None = None
    process_rss_bytes: int | None = None
    process_vsz_bytes: int | None = None
    process_count: int | None = None


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

    def container_stats(
        self,
        container_name: str,
        *,
        include_process_memory: bool = True,
    ) -> DockerContainerStatsSample:
        data = self._get_json(
            f"/containers/{_quote_container(container_name)}/stats?"
            "stream=false&one-shot=true"
        )
        sample = docker_stats_sample_from_dict(data)
        if include_process_memory:
            try:
                top = self.container_top(
                    container_name,
                    ps_args="-eo pid,ppid,rss,vsz,comm,args",
                )
                apply_process_memory_from_top(sample, top)
            except Exception:
                pass
        return sample

    def container_top(
        self,
        container_name: str,
        *,
        ps_args: str = "-eo pid,ppid,rss,vsz,comm,args",
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode({"ps_args": ps_args})
        return self._get_json(
            f"/containers/{_quote_container(container_name)}/top?{query}"
        )

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
    memory_detail_stats = memory_stats.get("stats") or {}
    pids_stats = data.get("pids_stats") or {}
    memory_usage_including_cache = _coerce_int(memory_stats.get("usage"))
    memory_inactive_file = _coerce_int(memory_detail_stats.get("inactive_file"))
    memory_cgroup_working_set = _container_memory_working_set(
        memory_usage_including_cache,
        memory_inactive_file,
    )
    return DockerContainerStatsSample(
        timestamp_monotonic=time.monotonic(),
        cpu_total_usage=_coerce_int(cpu_usage.get("total_usage")),
        system_cpu_usage=_coerce_int(cpu_stats.get("system_cpu_usage")),
        online_cpus=_coerce_int(cpu_stats.get("online_cpus")),
        memory_usage_bytes=memory_cgroup_working_set,
        memory_limit_bytes=_coerce_int(memory_stats.get("limit")),
        pids_current=_coerce_int(pids_stats.get("current")),
        memory_cgroup_working_set_bytes=memory_cgroup_working_set,
        memory_usage_including_cache_bytes=memory_usage_including_cache,
        memory_file_bytes=_coerce_int(
            memory_detail_stats.get("file", memory_detail_stats.get("cache"))
        ),
        memory_active_file_bytes=_coerce_int(memory_detail_stats.get("active_file")),
        memory_inactive_file_bytes=memory_inactive_file,
        memory_anon_bytes=_coerce_int(
            memory_detail_stats.get("anon", memory_detail_stats.get("rss"))
        ),
    )


def apply_process_memory_from_top(
    sample: DockerContainerStatsSample,
    top_data: dict[str, Any],
) -> DockerContainerStatsSample:
    memory = process_memory_from_top(top_data)
    if memory["process_count"] == 0:
        return sample
    sample.process_rss_bytes = memory["process_rss_bytes"]
    sample.process_vsz_bytes = memory["process_vsz_bytes"]
    sample.process_count = memory["process_count"]
    if sample.process_rss_bytes is not None:
        sample.memory_usage_bytes = sample.process_rss_bytes
    return sample


def process_memory_from_top(top_data: dict[str, Any]) -> dict[str, int | None]:
    titles = [str(title).upper() for title in top_data.get("Titles") or []]
    processes = top_data.get("Processes") or []
    rss_index = _first_title_index(titles, "RSS")
    vsz_index = _first_title_index(titles, "VSZ")
    rss_values = []
    vsz_values = []
    for process in processes:
        if not isinstance(process, list):
            continue
        rss_kib = _value_at(process, rss_index)
        vsz_kib = _value_at(process, vsz_index)
        rss = _coerce_int(rss_kib)
        vsz = _coerce_int(vsz_kib)
        if rss is not None:
            rss_values.append(rss * 1024)
        if vsz is not None:
            vsz_values.append(vsz * 1024)
    return {
        "process_rss_bytes": sum(rss_values) if rss_values else None,
        "process_vsz_bytes": sum(vsz_values) if vsz_values else None,
        "process_count": len(processes),
    }


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
    ram_including_cache_values = [
        sample.memory_usage_including_cache_bytes
        for sample in samples
        if sample.memory_usage_including_cache_bytes is not None
    ]
    ram_cgroup_working_set_values = [
        sample.memory_cgroup_working_set_bytes
        for sample in samples
        if sample.memory_cgroup_working_set_bytes is not None
    ]
    ram_file_values = [
        sample.memory_file_bytes
        for sample in samples
        if sample.memory_file_bytes is not None
    ]
    ram_anon_values = [
        sample.memory_anon_bytes
        for sample in samples
        if sample.memory_anon_bytes is not None
    ]
    process_rss_values = [
        sample.process_rss_bytes
        for sample in samples
        if sample.process_rss_bytes is not None
    ]
    process_vsz_values = [
        sample.process_vsz_bytes
        for sample in samples
        if sample.process_vsz_bytes is not None
    ]
    process_counts = [
        sample.process_count for sample in samples if sample.process_count is not None
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
        "container_ram_measurement_source": "docker_top_process_rss"
        if process_rss_values
        else "docker_cgroup_working_set",
        "container_process_rss_available": bool(process_rss_values),
        "container_ram_bytes_avg": int(round(_avg(ram_values)))
        if ram_values
        else None,
        "container_ram_bytes_peak": max(ram_values) if ram_values else None,
        "container_ram_working_set_bytes_avg": int(
            round(_avg(ram_cgroup_working_set_values))
        )
        if ram_cgroup_working_set_values
        else None,
        "container_ram_working_set_bytes_peak": max(ram_cgroup_working_set_values)
        if ram_cgroup_working_set_values
        else None,
        "container_ram_cgroup_working_set_bytes_avg": int(
            round(_avg(ram_cgroup_working_set_values))
        )
        if ram_cgroup_working_set_values
        else None,
        "container_ram_cgroup_working_set_bytes_peak": max(
            ram_cgroup_working_set_values
        )
        if ram_cgroup_working_set_values
        else None,
        "container_ram_including_cache_bytes_avg": int(
            round(_avg(ram_including_cache_values))
        )
        if ram_including_cache_values
        else None,
        "container_ram_including_cache_bytes_peak": max(ram_including_cache_values)
        if ram_including_cache_values
        else None,
        "container_ram_file_bytes_avg": int(round(_avg(ram_file_values)))
        if ram_file_values
        else None,
        "container_ram_file_bytes_peak": max(ram_file_values)
        if ram_file_values
        else None,
        "container_ram_anon_bytes_avg": int(round(_avg(ram_anon_values)))
        if ram_anon_values
        else None,
        "container_ram_anon_bytes_peak": max(ram_anon_values)
        if ram_anon_values
        else None,
        "container_process_rss_bytes_avg": int(round(_avg(process_rss_values)))
        if process_rss_values
        else None,
        "container_process_rss_bytes_peak": max(process_rss_values)
        if process_rss_values
        else None,
        "container_process_vsz_bytes_avg": int(round(_avg(process_vsz_values)))
        if process_vsz_values
        else None,
        "container_process_vsz_bytes_peak": max(process_vsz_values)
        if process_vsz_values
        else None,
        "container_process_count_peak": max(process_counts)
        if process_counts
        else None,
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


def _first_title_index(titles: list[str], title: str) -> int | None:
    try:
        return titles.index(title)
    except ValueError:
        return None


def _value_at(values: list[Any], index: int | None) -> Any:
    if index is None or index >= len(values):
        return None
    return values[index]


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _container_memory_working_set(
    usage_bytes: int | None,
    inactive_file_bytes: int | None,
) -> int | None:
    if usage_bytes is None:
        return None
    if inactive_file_bytes is None:
        return usage_bytes
    return max(0, usage_bytes - inactive_file_bytes)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
