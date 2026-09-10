import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from heracles_agents.cli.model_sweeps import expand_model_sweeps
from heracles_agents.experiment_definition import PipelineRegistry, register_pipeline
from heracles_agents.llm_interface import AnalyzedQuestions
from heracles_agents.pipelines.agentic_pipeline import d as agentic_pipeline_description
from heracles_agents.tool_calling.registry import ToolRegistry, register_tool
from heracles_agents.tools.cypher_query_tool import cypher_tool


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def base_template() -> dict:
    return {
        "dsg_interface": {"dsg_interface_type": "none"},
        "pipeline": "agentic",
        "phases": {
            "main": {
                "client": {"client_type": "openrouter"},
                "model_info": {"temperature": 0.2, "seed": 123},
                "agent_info": {
                    "prompt_settings": {
                        "base_prompt": "$HERACLES_AGENTS_PATH/examples/prompts/cypher/qa_agentic_cypher_prompt.yaml",
                        "output_type": "SLDP",
                        "answer_type_hint": True,
                    },
                    "tools": [
                        {
                            "name": "run_cypher_query",
                            "bound_args": {
                                "dsgdb_conf": {
                                    "dsg_interface_type": "heracles",
                                    "uri": "$HERACLES_NEO4J_URI",
                                }
                            },
                        }
                    ],
                    "tool_interface": "openrouter",
                    "max_iterations": 5,
                },
            }
        },
        "questions": "$HERACLES_AGENTS_PATH/examples/questions/qa_questions.yaml",
    }


def sweep_experiment(**sweep_overrides) -> dict:
    sweep = {
        "provider": "openrouter",
        "phase": "main",
        "configuration_name_template": "agentic-cypher-qa-{alias}",
        "result_configuration_name": "agentic-cypher-qa",
        "models": [
            {"alias": "enabled-model", "model": "provider/enabled", "enabled": True},
            {"alias": "disabled-model", "model": "provider/disabled", "enabled": False},
        ],
        "template": base_template(),
    }
    sweep.update(sweep_overrides)
    return {
        "metadata": {"benchmark_family": "test"},
        "model_sweeps": {"openrouter-cypher": sweep},
        "configurations": {},
    }


def test_no_model_sweeps_returns_experiment_unchanged(tmp_path):
    raw = {"metadata": {}, "configurations": {"manual": base_template()}}

    expanded, metadata = expand_model_sweeps(raw, tmp_path / "experiment.yaml")

    assert expanded == raw
    assert metadata == {}


def test_inline_models_expand_and_skip_disabled_models(tmp_path):
    raw = sweep_experiment()

    expanded, context = expand_model_sweeps(raw, tmp_path / "experiment.yaml")

    configurations = expanded["configurations"]
    assert list(configurations) == ["agentic-cypher-qa-enabled-model"]
    generated = configurations["agentic-cypher-qa-enabled-model"]
    phase = generated["phases"]["main"]
    assert phase["client"]["client_type"] == "openrouter"
    assert phase["model_info"]["model"] == "provider/enabled"
    assert phase["model_info"]["temperature"] == 0.2
    assert phase["model_info"]["seed"] == 123
    assert generated["pipeline"] == "agentic"
    assert generated["questions"].endswith("examples/questions/qa_questions.yaml")
    assert context == {
        "result_configuration_names": {
            "agentic-cypher-qa-enabled-model": "agentic-cypher-qa"
        }
    }
    assert "expanded_model_sweeps" not in expanded["metadata"]


def test_models_path_expands_to_configurations(tmp_path):
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"alias": "first", "model": "provider/first"},
                    {"alias": "second", "model": "provider/second"},
                ]
            }
        ),
        encoding="utf-8",
    )
    raw = sweep_experiment(models_path=str(models_path))
    del raw["model_sweeps"]["openrouter-cypher"]["models"]

    expanded, context = expand_model_sweeps(raw, tmp_path / "experiment.yaml")

    assert set(expanded["configurations"]) == {
        "agentic-cypher-qa-first",
        "agentic-cypher-qa-second",
    }
    assert context == {
        "result_configuration_names": {
            "agentic-cypher-qa-first": "agentic-cypher-qa",
            "agentic-cypher-qa-second": "agentic-cypher-qa",
        }
    }


def test_manual_configurations_are_preserved(tmp_path):
    raw = sweep_experiment()
    raw["configurations"]["manual-config"] = base_template()

    expanded, _ = expand_model_sweeps(raw, tmp_path / "experiment.yaml")

    assert set(expanded["configurations"]) == {
        "manual-config",
        "agentic-cypher-qa-enabled-model",
    }


def test_duplicate_generated_configuration_raises(tmp_path):
    raw = sweep_experiment()
    raw["configurations"]["agentic-cypher-qa-enabled-model"] = base_template()

    with pytest.raises(ValueError, match="already exists"):
        expand_model_sweeps(raw, tmp_path / "experiment.yaml")


def test_missing_phase_raises(tmp_path):
    raw = sweep_experiment(phase="missing")

    with pytest.raises(ValueError, match="target phase 'missing' is missing"):
        expand_model_sweeps(raw, tmp_path / "experiment.yaml")


def test_unsupported_provider_raises(tmp_path):
    raw = sweep_experiment(provider="openai")

    with pytest.raises(ValueError, match="Supported providers are: ollama, openrouter"):
        expand_model_sweeps(raw, tmp_path / "experiment.yaml")


def test_ollama_inline_models_expand_with_ollama_client_type(tmp_path):
    raw = sweep_experiment(provider="ollama")

    expanded, context = expand_model_sweeps(raw, tmp_path / "experiment.yaml")

    generated = expanded["configurations"]["agentic-cypher-qa-enabled-model"]
    assert generated["phases"]["main"]["client"]["client_type"] == "ollama"
    assert generated["phases"]["main"]["model_info"]["model"] == "provider/enabled"
    assert context["result_configuration_names"][
        "agentic-cypher-qa-enabled-model"
    ] == "agentic-cypher-qa"


def test_model_parameters_override_the_sweep_template(tmp_path):
    raw = sweep_experiment()
    raw["model_sweeps"]["openrouter-cypher"]["models"][0]["parameters"] = {
        "temperature": None,
        "seed": None,
        "reasoning": {"mode": "enabled", "effort": "xhigh"},
    }
    raw["model_sweeps"]["openrouter-cypher"]["models"][0]["capabilities"] = {
        "reasoning": "required",
        "reasoning_efforts": ["low", "medium", "high", "xhigh"],
        "temperature": True,
        "seed": True,
    }

    expanded, _ = expand_model_sweeps(raw, tmp_path / "experiment.yaml")

    model_info = expanded["configurations"][
        "agentic-cypher-qa-enabled-model"
    ]["phases"]["main"]["model_info"]
    assert model_info == {
        "model": "provider/enabled",
        "temperature": None,
        "seed": None,
        "reasoning": {"mode": "enabled", "effort": "xhigh"},
        "parameter_capabilities": {
            "reasoning": "required",
            "reasoning_efforts": ["low", "medium", "high", "xhigh"],
            "temperature": True,
            "seed": True,
        },
    }


def enabled_model_count(model_list_path: Path) -> int:
    data = yaml.safe_load(model_list_path.read_text(encoding="utf-8"))
    return sum(
        1
        for model in data["models"]
        if isinstance(model, dict) and model.get("enabled", True) is not False
    )


def test_real_openrouter_sweep_files_expand_to_enabled_valid_configurations():
    if "agentic" not in PipelineRegistry.pipelines:
        register_pipeline(agentic_pipeline_description)
    if "run_cypher_query" not in ToolRegistry.tools:
        register_tool(cypher_tool)

    env = {
        "HERACLES_AGENTS_PATH": str(project_root()),
        "HERACLES_OPENROUTER_API_KEY": "test-key",
        "HERACLES_NEO4J_USERNAME": "neo4j",
        "HERACLES_NEO4J_PASSWORD": "password",
        "HERACLES_NEO4J_URI": "neo4j://localhost:7687",
    }
    paths = [
        project_root() / "examples/experiments/openrouter/cypher_model_sweep.yaml",
        project_root() / "examples/experiments/openrouter/pddl_model_sweep.yaml",
    ]
    expected_count = enabled_model_count(
        project_root() / "examples/experiments/openrouter/model_lists/example.yaml"
    )

    with (
        patch.dict(os.environ, env, clear=False),
        patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ),
    ):
        runner_path = project_root() / "examples/experiment_runner.py"
        spec = importlib.util.spec_from_file_location("experiment_runner", runner_path)
        experiment_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment_runner)
        loaded = [experiment_runner.load_experiment_with_context(path) for path in paths]

    for experiment, _context in loaded:
        assert len(experiment.configurations) == expected_count
        assert "expanded_model_sweeps" not in experiment.metadata

    experiments = [experiment for experiment, _context in loaded]
    contexts = [context for _experiment, context in loaded]
    assert contexts[0]["result_configuration_names"][
        "agentic-cypher-qa-deepseek-v4-flash"
    ] == "agentic-cypher-qa"
    assert contexts[1]["result_configuration_names"][
        "agentic-cypher-pddl-deepseek-v4-flash"
    ] == "agentic-cypher-pddl"

    assert (
        experiments[0]
        .configurations["agentic-cypher-qa-deepseek-v4-flash"]
        .phases["main"]
        .model_info
        .model
        == "deepseek/deepseek-v4-flash"
    )
    assert (
        experiments[1]
        .configurations["agentic-cypher-pddl-deepseek-v4-flash"]
        .phases["main"]
        .model_info
        .model
        == "deepseek/deepseek-v4-flash"
    )


def test_real_ollama_sweep_files_expand_to_enabled_valid_configurations():
    if "agentic" not in PipelineRegistry.pipelines:
        register_pipeline(agentic_pipeline_description)
    if "run_cypher_query" not in ToolRegistry.tools:
        register_tool(cypher_tool)

    env = {
        "HERACLES_AGENTS_PATH": str(project_root()),
        "OLLAMA_HOST": "http://ollama:11434",
        "HERACLES_NEO4J_USERNAME": "neo4j",
        "HERACLES_NEO4J_PASSWORD": "password",
        "HERACLES_NEO4J_URI": "neo4j://localhost:7687",
    }
    paths = [
        project_root() / "examples/experiments/ollama/cypher_model_sweep.yaml",
        project_root() / "examples/experiments/ollama/pddl_model_sweep.yaml",
    ]
    expected_count = enabled_model_count(
        project_root() / "examples/experiments/ollama/model_lists/example.yaml"
    )

    with (
        patch.dict(os.environ, env, clear=False),
        patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ),
    ):
        runner_path = project_root() / "examples/experiment_runner.py"
        spec = importlib.util.spec_from_file_location("experiment_runner", runner_path)
        experiment_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment_runner)
        loaded = [experiment_runner.load_experiment_with_context(path) for path in paths]

    for experiment, _context in loaded:
        assert len(experiment.configurations) == expected_count
        assert "expanded_model_sweeps" not in experiment.metadata

    experiments = [experiment for experiment, _context in loaded]
    contexts = [context for _experiment, context in loaded]
    assert contexts[0]["result_configuration_names"][
        "agentic-cypher-qa-gemma-4-12b"
    ] == "agentic-cypher-qa"
    assert contexts[1]["result_configuration_names"][
        "agentic-cypher-pddl-gemma-4-12b"
    ] == "agentic-cypher-pddl"

    assert (
        experiments[0]
        .configurations["agentic-cypher-qa-gemma-4-12b"]
        .phases["main"]
        .client
        .client_type
        == "ollama"
    )
    assert (
        experiments[1]
        .configurations["agentic-cypher-pddl-gemma-4-26b"]
        .phases["main"]
        .model_info
        .model
        == "gemma4:26b"
    )


def test_sweep_run_writes_one_result_file_per_completed_configuration(
    tmp_path,
    monkeypatch,
):
    runner_path = project_root() / "examples/experiment_runner.py"
    spec = importlib.util.spec_from_file_location("experiment_runner", runner_path)
    experiment_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_runner)

    experiment = SimpleNamespace(
        metadata={"task": "cypher"},
        configurations={
            "config-a": SimpleNamespace(
                pipeline=SimpleNamespace(
                    function=lambda _config: AnalyzedQuestions(analyzed_questions=[]),
                ),
                phases={
                    "main": SimpleNamespace(
                        client=SimpleNamespace(client_type="openrouter"),
                        model_info=SimpleNamespace(model="provider/model-a"),
                    )
                },
            ),
            "config-b": SimpleNamespace(
                pipeline=SimpleNamespace(
                    function=lambda _config: AnalyzedQuestions(analyzed_questions=[]),
                ),
                phases={
                    "main": SimpleNamespace(
                        client=SimpleNamespace(client_type="openrouter"),
                        model_info=SimpleNamespace(model="provider/model-b"),
                    )
                },
            ),
        },
    )
    monkeypatch.setattr(
        experiment_runner,
        "load_experiment_with_context",
        lambda _path: (
            experiment,
            {"result_configuration_names": {"config-a": "agentic-cypher-qa"}},
        ),
    )

    result_paths = experiment_runner.run_experiment(
        project_root() / "examples/experiments/openrouter/cypher_model_sweep.yaml",
        tmp_path,
        {"config-a"},
        continue_on_error=False,
        display_results=False,
    )

    assert result_paths == [
        tmp_path
        / "openrouter"
        / "cypher_model_sweep"
        / "config-a_results.yaml"
    ]
    result = yaml.safe_load(result_paths[0].read_text(encoding="utf-8"))
    assert list(result["experiment_configurations"]) == ["agentic-cypher-qa"]
    assert "expanded_model_sweeps" not in result["metadata"]
    assert result["metadata"]["llm_configurations"] == {
        "agentic-cypher-qa": {
            "phases": {
                    "main": {
                        "provider": "openrouter",
                        "model_identifier": "provider/model-a",
                        "request_parameters": {
                            "temperature": None,
                            "seed": None,
                            "reasoning": None,
                        },
                    }
            }
        }
    }


def test_experiment_runner_debug_configures_local_metrics_sample_output():
    runner_path = project_root() / "examples/experiment_runner.py"
    spec = importlib.util.spec_from_file_location("experiment_runner", runner_path)
    experiment_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_runner)
    experiment_config = SimpleNamespace(
        local_metrics={
            "enabled": True,
            "sample_interval_seconds": 0.025,
        }
    )
    result_path = Path("output/ollama/canary_results.yaml")

    experiment_runner.configure_debug_outputs(
        experiment_config,
        result_path,
        "canary-local-metrics",
    )

    assert experiment_config.local_metrics["record_samples"] is True
    assert (
        experiment_config.local_metrics["samples_output_path"]
        == "output/ollama/canary_canary-local-metrics_local_metrics_samples.yaml"
    )
