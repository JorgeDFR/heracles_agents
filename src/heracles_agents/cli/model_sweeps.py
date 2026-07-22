"""Expand compact model sweep YAML sections into experiment configurations."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def expand_model_sweeps(
    raw_experiment: dict[str, Any], source_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand top-level model_sweeps into ordinary experiment configurations.

    The returned context is for the runner only. It is intentionally not written
    to result metadata.
    """

    if not isinstance(raw_experiment, dict):
        raise ValueError("Experiment data must be a mapping.")

    if "model_sweeps" not in raw_experiment:
        return copy.deepcopy(raw_experiment), {}

    expanded_experiment = copy.deepcopy(raw_experiment)
    raw_sweeps = expanded_experiment.pop("model_sweeps")
    if not isinstance(raw_sweeps, dict):
        raise ValueError("'model_sweeps' must be a mapping.")

    configurations = expanded_experiment.setdefault("configurations", {})
    if not isinstance(configurations, dict):
        raise ValueError("'configurations' must be a mapping.")

    sweep_context: dict[str, Any] = {"result_configuration_names": {}}
    for sweep_name, sweep_config in raw_sweeps.items():
        if not isinstance(sweep_config, dict):
            raise ValueError(f"Model sweep '{sweep_name}' must be a mapping.")

        provider = sweep_config.get("provider")
        if provider != "openrouter":
            raise ValueError(
                f"Model sweep '{sweep_name}' uses provider '{provider}'. "
                "Only provider 'openrouter' is supported."
            )

        phase = _required_str(sweep_config, "phase", sweep_name)
        name_template = _required_str(
            sweep_config, "configuration_name_template", sweep_name
        )
        result_configuration_name = sweep_config.get("result_configuration_name")
        if result_configuration_name is not None and (
            not isinstance(result_configuration_name, str)
            or not result_configuration_name
        ):
            raise ValueError(
                f"Model sweep '{sweep_name}' field 'result_configuration_name' "
                "must be a non-empty string when provided."
            )
        template = sweep_config.get("template")
        if not isinstance(template, dict):
            raise ValueError(f"Model sweep '{sweep_name}' requires a mapping template.")

        models_path = sweep_config.get("models_path")
        inline_models = sweep_config.get("models")
        if (models_path is None) == (inline_models is None):
            raise ValueError(
                f"Model sweep '{sweep_name}' must define exactly one of "
                "'models' or 'models_path'."
            )

        models, _ = _load_models(
            inline_models,
            models_path,
            source_path,
            sweep_name,
        )

        for model_entry in models:
            alias = _required_str(model_entry, "alias", sweep_name)
            model = _required_str(model_entry, "model", sweep_name)
            if model_entry.get("enabled", True) is False:
                continue

            configuration_name = name_template.format(alias=alias, model=model)
            output_configuration_name = result_configuration_name or configuration_name
            if configuration_name in configurations:
                raise ValueError(
                    f"Generated configuration '{configuration_name}' from model "
                    f"sweep '{sweep_name}' already exists."
                )

            generated_config = copy.deepcopy(template)
            phases = generated_config.get("phases")
            if not isinstance(phases, dict) or phase not in phases:
                raise ValueError(
                    f"Model sweep '{sweep_name}' target phase '{phase}' is missing "
                    "from the template."
                )

            phase_config = phases[phase]
            if not isinstance(phase_config, dict):
                raise ValueError(
                    f"Model sweep '{sweep_name}' phase '{phase}' must be a mapping."
                )

            client_config = phase_config.setdefault("client", {})
            if not isinstance(client_config, dict):
                raise ValueError(
                    f"Model sweep '{sweep_name}' phase '{phase}' client must be a "
                    "mapping."
                )
            client_config["client_type"] = "openrouter"

            model_info = phase_config.setdefault("model_info", {})
            if not isinstance(model_info, dict):
                raise ValueError(
                    f"Model sweep '{sweep_name}' phase '{phase}' model_info must be "
                    "a mapping."
                )
            model_info["model"] = model

            configurations[configuration_name] = generated_config
            sweep_context["result_configuration_names"][
                configuration_name
            ] = output_configuration_name

    return expanded_experiment, sweep_context


def _load_models(
    inline_models: Any,
    models_path: Any,
    source_path: Path,
    sweep_name: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    if models_path is not None:
        resolved_path = _resolve_models_path(models_path, source_path)
        with resolved_path.open("r", encoding="utf-8") as fo:
            data = yaml.safe_load(fo)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError(
                f"Model list for sweep '{sweep_name}' must contain a 'models' list."
            )
        return _validate_models(data["models"], sweep_name), resolved_path

    if not isinstance(inline_models, list):
        raise ValueError(f"Inline models for sweep '{sweep_name}' must be a list.")
    return _validate_models(inline_models, sweep_name), None


def _resolve_models_path(models_path: Any, source_path: Path) -> Path:
    path = Path(os.path.expandvars(str(models_path))).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()


def _validate_models(models: list[Any], sweep_name: str) -> list[dict[str, Any]]:
    validated = []
    for idx, model_entry in enumerate(models):
        if not isinstance(model_entry, dict):
            raise ValueError(
                f"Model entry {idx} in sweep '{sweep_name}' must be a mapping."
            )
        validated.append(model_entry)
    return validated


def _required_str(mapping: dict[str, Any], key: str, sweep_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Model sweep '{sweep_name}' requires string field '{key}'.")
    return value
