import yaml
from rich.console import Console

from heracles_agents.cli.result_reporting import (
    load_result_file,
    load_result_files,
    render_html_report,
    render_terminal_summary,
    summarize_configuration,
)


def sample_result_data(*, answer="<answer>2</answer>"):
    return {
        "metadata": {
            "source_experiment": "examples/experiments/openrouter/canary.yaml",
            "elapsed_seconds": 12.5,
            "failed_configurations": {},
            "llm_configurations": {
                "canary": {
                    "phases": {
                        "main": {
                            "provider": "openrouter",
                            "model_identifier": "openai/gpt-test",
                        }
                    }
                }
            },
        },
        "experiment_configurations": {
            "canary": {
                "local_resources": {
                    "measurement_scope": "ollama_container",
                    "ollama_container_name": "ollama",
                    "sample_interval_seconds": 0.025,
                    "baseline_seconds": 5,
                    "baseline_adjusted": True,
                    "telemetry": {
                        "gpu_effective_interval_seconds_avg": 0.026,
                        "docker_effective_interval_seconds_avg": 1.0,
                    },
                    "cpu": {
                        "container_cpu_percent_avg": 240.0,
                        "container_cpu_percent_peak": 320.0,
                    },
                    "ram": {
                        "container_ram_bytes_peak": 1048576,
                    },
                    "gpu": {
                        "gpu_memory_used_mib_adjusted_peak": 1024,
                        "gpu_utilization_percent_adjusted_avg": 75,
                        "gpu_power_w_adjusted_avg": 100,
                        "gpu_energy_wh_adjusted": 0.01,
                    },
                    "ollama": {
                        "output_tokens_per_second": 20,
                        "load_duration_seconds": 1.5,
                    },
                    "warnings": [
                        "GPU metrics are baseline-adjusted.",
                    ],
                },
                "cost_summary": {
                    "total_cost_usd": 0.003,
                    "cost_per_question_usd": 0.0015,
                    "cost_per_correct_answer_usd": 0.003,
                    "currency": "USD",
                    "cost_basis": "provider_reported",
                },
                "analyzed_questions": [
                    {
                        "question": {
                            "name": "Arithmetic <One>",
                            "question": "What is 1 + 1?",
                            "solution": "2",
                            "uid": 1,
                            "correctness_comparator": {
                                "comparison_type": "SLDP",
                                "relation": "equal",
                            },
                        },
                        "answer": answer,
                        "analysis": {
                            "valid_answer_format": True,
                            "correct": True,
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "n_tool_calls": 0,
                            "latency": {
                                "end_to_end_seconds": 2.0,
                                "llm_call_seconds": 1.5,
                                "tool_execution_seconds": 0.1,
                                "neo4j_query_seconds": 0.2,
                                "parsing_validation_seconds": 0.3,
                                "retry_wait_seconds": 0.4,
                                "time_to_first_token_seconds": None,
                            },
                            "cost": {
                                "total_cost_usd": 0.001,
                                "cost_basis": "provider_reported",
                                "currency": "USD",
                                "llm_calls": [],
                            },
                            "local_resources": {
                                "measurement_scope": "ollama_container",
                                "ollama_container_name": "ollama",
                                "sample_interval_seconds": 0.5,
                                "baseline_seconds": 5,
                                "baseline_adjusted": True,
                                "cpu": {
                                    "container_cpu_percent_avg": 240.0,
                                    "container_cpu_percent_peak": 320.0,
                                },
                                "ram": {
                                    "container_ram_bytes_peak": 1048576,
                                },
                                "gpu": {
                                    "gpu_memory_used_mib_adjusted_peak": 1024,
                                    "gpu_utilization_percent_adjusted_avg": 75,
                                    "gpu_power_w_adjusted_avg": 100,
                                    "gpu_energy_wh_adjusted": 0.01,
                                },
                                "ollama": {
                                    "output_tokens_per_second": 20,
                                    "load_duration_seconds": 1.5,
                                },
                                "warnings": [
                                    "GPU metrics are baseline-adjusted.",
                                ],
                            },
                        },
                        "completed": True,
                        "sequences": [
                            {
                                "description": "main",
                                "responses": [
                                    {
                                        "raw_response": "raw response",
                                        "parsed_response": "parsed response",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "question": {
                            "name": "Failed",
                            "question": "What is 2 + 2?",
                            "solution": "4",
                            "uid": 2,
                            "correctness_comparator": {
                                "comparison_type": "SLDP",
                                "relation": "equal",
                            },
                        },
                        "answer": "5",
                        "analysis": {
                            "valid_answer_format": True,
                            "correct": False,
                            "input_tokens": 20,
                            "output_tokens": 6,
                            "n_tool_calls": 1,
                        },
                        "completed": True,
                        "sequences": [],
                    },
                ],
            }
        },
    }


def write_yaml(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_result_file_normalizes_experiment_yaml_shape(tmp_path):
    path = write_yaml(tmp_path, "results.yaml", sample_result_data())

    source = load_result_file(path)

    assert source.path == path.resolve()
    assert source.metadata["elapsed_seconds"] == 12.5
    assert len(source.configurations) == 1
    configuration = source.configurations[0]
    assert configuration.configuration_name == "canary"
    assert configuration.provider_models[0].provider == "openrouter"
    assert configuration.provider_models[0].model_identifier == "openai/gpt-test"
    assert configuration.local_resources["ollama"]["load_duration_seconds"] == 1.5
    assert configuration.questions[0].name == "Arithmetic <One>"
    assert configuration.questions[0].latency["end_to_end_seconds"] == 2.0
    assert configuration.questions[0].cost["total_cost_usd"] == 0.001
    assert configuration.questions[0].n_sequences == 1


def test_load_result_file_supports_single_analyzed_questions_shape(tmp_path):
    data = {
        "analyzed_questions": sample_result_data()["experiment_configurations"][
            "canary"
        ]["analyzed_questions"]
    }
    path = write_yaml(tmp_path, "single.yaml", data)

    source = load_result_file(path)

    assert len(source.configurations) == 1
    assert source.configurations[0].configuration_name == "results"
    assert len(source.configurations[0].questions) == 2


def test_summarize_configuration_counts_core_metrics_and_optional_metrics(tmp_path):
    source = load_result_file(write_yaml(tmp_path, "results.yaml", sample_result_data()))

    summary = summarize_configuration(source.configurations[0].questions)

    assert summary.questions == 2
    assert summary.completed_count == 2
    assert summary.correct_count == 1
    assert summary.accuracy == 0.5
    assert summary.input_tokens_total == 30
    assert summary.input_tokens_avg == 15
    assert summary.output_tokens_total == 10
    assert summary.tool_calls_total == 1
    assert summary.end_to_end_latency_total == 2.0
    assert summary.end_to_end_latency_avg == 2.0
    assert summary.end_to_end_latency_p50 == 2.0
    assert summary.end_to_end_latency_p95 == 2.0
    assert summary.llm_latency_total == 1.5
    assert summary.tool_latency_total == 0.1
    assert summary.neo4j_latency_total == 0.2
    assert summary.validation_latency_total == 0.3
    assert summary.retry_wait_total == 0.4
    assert summary.cost_total_usd == 0.001
    assert summary.cost_per_question_usd == 0.0005
    assert summary.cost_per_correct_answer_usd == 0.001


def test_terminal_summary_default_columns_include_question_answer_and_exclude_details(
    tmp_path,
):
    source = load_result_file(write_yaml(tmp_path, "results.yaml", sample_result_data()))
    console = Console(record=True, width=240)

    render_terminal_summary([source], console=console)
    output = console.export_text()

    assert "Topic" in output
    assert "Question" in output
    assert "Solution" in output
    assert "Answer" in output
    assert "What is 1 + 1?" in output
    assert "End-to-End" not in output
    assert "Cost USD" not in output
    assert "Sequences" not in output


def test_terminal_summary_can_include_sequence_count(tmp_path):
    source = load_result_file(write_yaml(tmp_path, "results.yaml", sample_result_data()))
    console = Console(record=True, width=240)

    render_terminal_summary([source], console=console, show_sequences=True)
    output = console.export_text()

    assert "Sequences" in output


def test_html_report_contains_metrics_sequences_and_escaped_data(tmp_path):
    source = load_result_file(
        write_yaml(
            tmp_path,
            "results.yaml",
            sample_result_data(answer="<script>alert('x')</script>"),
        )
    )
    output_path = tmp_path / "report.html"

    render_html_report([source], output_path)
    html = output_path.read_text(encoding="utf-8")

    assert "Heracles Experiment Results" in html
    assert 'label: "End-to-End Average Latency"' in html
    assert "Latency (s)" in html
    assert 'label: "End-to-End Latency"' in html
    assert 'label: "End-to-End Latency (s)"' not in html
    assert "function seconds(value)" in html
    assert "function percent(value)" in html
    assert "hasMeaningfulValue" in html
    assert "Cost USD" in html
    assert "Local Resources" in html
    assert 'label: "CPU Avg"' in html
    assert 'label: "GPU Avg"' in html
    assert 'label: "VRAM Peak"' in html
    assert 'label: "Throughput"' in html
    assert 'label: "Load Time"' in html
    assert 'label: "Docker Sample"' not in html
    assert 'label: "GPU Sample"' not in html
    assert "tok/s" in html
    assert '"local_resources"' in html
    assert '"local_resources_summary"' in html
    assert 'label: "Topic"' in html
    assert "Topic: ${escapeHtml(q.name)}" in html
    assert "Sequences" in html
    assert "openrouter" in html
    assert "grid-template-columns: repeat(5, minmax(0, 1fr))" in html
    assert "grid-column: 1 / -1" in html
    assert "white-space: nowrap" in html
    assert '<details class="report-section" open>' in html
    assert "function filterContext()" in html
    assert "const context = filterContext();" in html
    assert "top: 100px" not in html
    assert "setExclusiveGroupSelection" in html
    assert 'data-group="quality" checked' in html
    assert 'data-group="tokens" checked' not in html
    assert "<dt>Parsed</dt>" not in html
    assert "parsed response" in html
    assert "<script>alert('x')</script>" not in html
    assert "\\u003cscript>alert('x')\\u003c/script>" in html


def test_load_result_files_supports_multiple_yaml_files(tmp_path):
    first = write_yaml(tmp_path, "first.yaml", sample_result_data())
    second = write_yaml(tmp_path, "second.yaml", sample_result_data())

    sources = load_result_files([first, second])

    assert [source.path for source in sources] == [first.resolve(), second.resolve()]
