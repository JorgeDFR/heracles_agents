"""NVIDIA GPU telemetry through nvidia-smi."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GpuStatsSample:
    timestamp_monotonic: float
    gpu_index: int
    name: str
    memory_used_mib: float | None
    memory_total_mib: float | None
    utilization_gpu_percent: float | None
    utilization_memory_percent: float | None
    power_draw_w: float | None


GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
    "--format=csv,noheader,nounits",
]

def sample_gpu_stats() -> list[GpuStatsSample]:
    output = subprocess.check_output(
        GPU_QUERY,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=5,
    )
    return parse_gpu_stats(output)


def parse_gpu_stats(output: str) -> list[GpuStatsSample]:
    timestamp = time.monotonic()
    samples = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        samples.append(
            GpuStatsSample(
                timestamp_monotonic=timestamp,
                gpu_index=int(parts[0]),
                name=parts[1],
                memory_used_mib=_coerce_float(parts[2]),
                memory_total_mib=_coerce_float(parts[3]),
                utilization_gpu_percent=_coerce_float(parts[4]),
                utilization_memory_percent=_coerce_float(parts[5]),
                power_draw_w=_coerce_float(parts[6]),
            )
        )
    return samples


def summarize_gpu_samples(
    samples: list[GpuStatsSample],
    baseline_samples: list[GpuStatsSample],
    *,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    raw_memory = _values(samples, "memory_used_mib")
    raw_util = _values(samples, "utilization_gpu_percent")
    raw_power = _values(samples, "power_draw_w")
    baseline_memory = _values(baseline_samples, "memory_used_mib")
    baseline_util = _values(baseline_samples, "utilization_gpu_percent")
    baseline_power = _values(baseline_samples, "power_draw_w")
    baseline_memory_avg = _avg(baseline_memory)
    baseline_util_avg = _avg(baseline_util)
    baseline_power_avg = _avg(baseline_power)

    adjusted_memory = _adjusted_values(raw_memory, baseline_memory_avg)
    adjusted_util = _adjusted_values(raw_util, baseline_util_avg)
    adjusted_power = _adjusted_values(raw_power, baseline_power_avg)
    sample_intervals = _sample_intervals(samples, sample_interval_seconds)
    return {
        "gpu_memory_used_mib_raw_avg": _avg(raw_memory),
        "gpu_memory_used_mib_raw_peak": max(raw_memory) if raw_memory else None,
        "gpu_memory_used_mib_baseline_avg": baseline_memory_avg,
        "gpu_memory_used_mib_adjusted_avg": _avg(adjusted_memory),
        "gpu_memory_used_mib_adjusted_peak": max(adjusted_memory)
        if adjusted_memory
        else None,
        "gpu_utilization_percent_raw_avg": _avg(raw_util),
        "gpu_utilization_percent_baseline_avg": baseline_util_avg,
        "gpu_utilization_percent_adjusted_avg": _avg(adjusted_util),
        "gpu_power_w_raw_avg": _avg(raw_power),
        "gpu_power_w_baseline_avg": baseline_power_avg,
        "gpu_power_w_adjusted_avg": _avg(adjusted_power),
        "gpu_energy_wh_adjusted": round(
            sum(
                power * interval
                for power, interval in zip(adjusted_power, sample_intervals)
            )
            / 3600.0,
            9,
        )
        if adjusted_power
        else None,
        "gpu_sample_count": len(samples),
        "gpu_effective_sample_interval_seconds_avg": _avg(sample_intervals)
        if len(samples) > 1
        else None,
    }


def _values(samples, attr: str) -> list[float]:
    return [
        value
        for value in (getattr(sample, attr, None) for sample in samples)
        if value is not None
    ]


def _adjusted_values(values: list[float], baseline: float | None) -> list[float]:
    if baseline is None:
        return []
    return [max(0.0, value - baseline) for value in values]


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _sample_intervals(
    samples: list[GpuStatsSample],
    fallback_interval_seconds: float,
) -> list[float]:
    if not samples:
        return []
    timestamps = [sample.timestamp_monotonic for sample in samples]
    intervals = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current >= previous
    ]
    fallback = _avg(intervals) or fallback_interval_seconds
    return intervals + [fallback]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
