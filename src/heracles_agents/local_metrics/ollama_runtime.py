"""Ollama API and response runtime metric helpers."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


def list_loaded_models(ollama_host: str) -> list[dict[str, Any]]:
    data = _ollama_json(ollama_host, "/api/ps")
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, list) else []


def unload_loaded_models(ollama_host: str) -> list[str]:
    unloaded = []
    for model in list_loaded_models(ollama_host):
        model_name = model.get("name") or model.get("model")
        if not model_name:
            continue
        payload = {
            "model": model_name,
            "prompt": "",
            "keep_alive": 0,
            "stream": False,
        }
        _ollama_json(ollama_host, "/api/generate", payload=payload)
        unloaded.append(model_name)
    return unloaded


def extract_ollama_response_metrics(response: Any) -> dict[str, Any]:
    total_duration = _duration_seconds(_get_attr_or_key(response, "total_duration"))
    load_duration = _duration_seconds(_get_attr_or_key(response, "load_duration"))
    prompt_eval_count = _coerce_int(_get_attr_or_key(response, "prompt_eval_count"))
    prompt_eval_duration = _duration_seconds(
        _get_attr_or_key(response, "prompt_eval_duration")
    )
    eval_count = _coerce_int(_get_attr_or_key(response, "eval_count"))
    eval_duration = _duration_seconds(_get_attr_or_key(response, "eval_duration"))
    return {
        "total_duration_seconds": total_duration,
        "load_duration_seconds": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_seconds": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration,
        "prompt_tokens_per_second": _rate(prompt_eval_count, prompt_eval_duration),
        "output_tokens_per_second": _rate(eval_count, eval_duration),
        "steady_state_seconds": round(
            (prompt_eval_duration or 0.0) + (eval_duration or 0.0),
            6,
        )
        if prompt_eval_duration is not None or eval_duration is not None
        else None,
    }


def summarize_ollama_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    keys = [
        "total_duration_seconds",
        "load_duration_seconds",
        "prompt_eval_count",
        "prompt_eval_duration_seconds",
        "eval_count",
        "eval_duration_seconds",
        "prompt_tokens_per_second",
        "output_tokens_per_second",
        "steady_state_seconds",
    ]
    summary = {}
    for key in keys:
        values = [_coerce_float(metric.get(key)) for metric in metrics]
        known = [value for value in values if value is not None]
        if not known:
            summary[key] = None
        elif key.endswith("_count"):
            summary[key] = int(sum(known))
        else:
            summary[key] = round(sum(known), 6)

    total_prompt_count = _sum_known(metrics, "prompt_eval_count")
    total_prompt_duration = _sum_known(metrics, "prompt_eval_duration_seconds")
    total_eval_count = _sum_known(metrics, "eval_count")
    total_eval_duration = _sum_known(metrics, "eval_duration_seconds")
    summary["prompt_tokens_per_second"] = _rate(
        total_prompt_count,
        total_prompt_duration,
    )
    summary["output_tokens_per_second"] = _rate(total_eval_count, total_eval_duration)
    summary["llm_call_count"] = len(metrics)
    return summary


def _ollama_json(
    ollama_host: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    url = urllib.parse.urljoin(ollama_host.rstrip("/") + "/", path.lstrip("/"))
    body = None
    headers = {}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read().decode("utf-8")
    return json.loads(data) if data else {}


def _duration_seconds(value: Any) -> float | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return round(number / 1_000_000_000.0, 6)


def _rate(count: int | float | None, duration_seconds: float | None) -> float | None:
    if count is None or duration_seconds is None or duration_seconds <= 0:
        return None
    return round(float(count) / duration_seconds, 6)


def _sum_known(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [_coerce_float(metric.get(key)) for metric in metrics]
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _get_attr_or_key(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

