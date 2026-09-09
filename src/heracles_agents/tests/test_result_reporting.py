import yaml
from rich.console import Console

from heracles_agents.cli.result_reporting import (
    _html_payload,
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
    source = load_result_file(
        write_yaml(tmp_path, "results.yaml", sample_result_data())
    )

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
    source = load_result_file(
        write_yaml(tmp_path, "results.yaml", sample_result_data())
    )
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
    source = load_result_file(
        write_yaml(tmp_path, "results.yaml", sample_result_data())
    )
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
    assert 'label: "Average End-to-End Latency"' in html
    assert 'label: "Cost per Success"' in html
    assert "Latency (s)" in html
    assert 'label: "End-to-End Latency"' in html
    assert 'label: "End-to-End Latency (s)"' not in html
    assert "function seconds(value)" in html
    assert "function percent(value)" in html
    assert "function compareSortValues" in html
    assert "function applyColumnFilters(table)" in html
    assert "hasMeaningfulValue" in html
    assert 'class="column-filter"' in html
    assert 'class="filter-row"' in html
    assert 'data-sort-index="${index}"' in html
    assert "sort-asc" in html
    assert "sort-desc" in html
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
    assert "Messages" in html
    assert 'class="message-card"' in html
    assert 'class="message-role"' in html
    assert 'class="sequence-group"' in html
    assert "Raw Response" in html
    assert "Parsed Response" not in html
    assert "openrouter" in html
    assert '"provider": "openrouter"' in html
    assert '"providers": ["openrouter"]' in html
    assert html.index(">Provider Models</button>") < html.index(">Overview</button>")
    assert html.index(">Overview</button>") < html.index(">Questions</button>")
    assert html.index(">Questions</button>") < html.index(">Artifacts</button>")
    assert 'id="providers-tab" class="tab-panel active"' in html
    assert 'id="overview-tab" class="tab-panel"' in html
    assert 'id="questions-tab" class="tab-panel"' in html
    assert 'id="artifacts-tab" class="tab-panel"' in html
    assert "function selectTab(button)" in html
    assert "function renderArtifacts()" in html
    assert 'class="metric-controls" aria-label="Metric columns" hidden' in html
    assert '!["overview-tab", "questions-tab"].includes(panelId)' in html
    assert "function rowsForActiveMetricGroup(rows, tableKind)" in html
    assert 'rowsForActiveMetricGroup(report.overview, "overview")' in html
    assert 'rowsForActiveMetricGroup(report.questions, "questions")' in html
    assert "sourceFilter" not in html
    assert "providerFilter" not in html
    assert "searchInput" not in html
    assert "white-space: nowrap" in html
    assert "report-section" not in html
    assert "top: 100px" not in html
    assert 'type="radio" name="metric-group" data-group="quality" checked' in html
    assert 'type="radio" name="metric-group" data-group="tokens"' in html
    assert 'class="status-pill status-good"' in html
    assert 'class="status-pill status-bad"' in html
    assert "h2 { font-size: 20px; margin: 0 0 12px; }" in html
    assert "parsed response" in html
    assert "<script>alert('x')</script>" not in html
    assert "\\u003cscript>alert('x')\\u003c/script>" in html


def test_html_payload_structures_saved_responses_as_messages(tmp_path):
    data = sample_result_data()
    responses = data["experiment_configurations"]["canary"]["analyzed_questions"][0][
        "sequences"
    ][0]["responses"]
    responses[:] = [
        {
            "raw_response": repr(
                {
                    "role": "assistant",
                    "content": "Checking the graph",
                    "tool_calls": [
                        {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (n) RETURN n"},
                        }
                    ],
                    "request_id": "request-1",
                }
            ),
            "parsed_response": "Checking the graph",
        },
        {
            "raw_response": repr(
                {
                    "role": "tool",
                    "tool_name": "run_cypher_query",
                    "content": [{"n": "O1"}],
                }
            ),
            "parsed_response": "tool: [{'n': 'O1'}]",
        },
        {
            "raw_response": (
                "role='assistant' content='' thinking=None images=None "
                "tool_name=None tool_calls=[ToolCall(function=Function("
                "name='run_cypher_query', arguments={'cypher_string': "
                "'MATCH (n) RETURN n'}))]"
            ),
            "parsed_response": "Preparing the query",
        },
    ]
    source = load_result_file(write_yaml(tmp_path, "results.yaml", data))

    messages = _html_payload([source])["questions"][0]["sequences"][0]["messages"]

    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "Checking the graph"
    assert messages[0]["tool_calls"][0]["name"] == "run_cypher_query"
    assert messages[0]["metadata"] == {"request_id": "request-1"}
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_name"] == "run_cypher_query"
    assert messages[1]["content"] == [{"n": "O1"}]
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Preparing the query"
    assert messages[2]["tool_calls"] == [
        {
            "function": {
                "name": "run_cypher_query",
                "arguments": {"cypher_string": "MATCH (n) RETURN n"},
            }
        }
    ]


def test_html_payload_collects_result_and_referenced_artifacts(tmp_path):
    prompt_path = write_yaml(tmp_path, "prompt.yaml", {"system": "Be concise."})
    experiment_path = write_yaml(
        tmp_path,
        "experiment.yaml",
        {"template": {"base_prompt": str(prompt_path)}},
    )
    data = sample_result_data()
    data["metadata"]["source_experiment"] = str(experiment_path)
    result_path = write_yaml(tmp_path, "results.yaml", data)
    source = load_result_file(result_path)

    artifacts = _html_payload([source])["artifacts"]

    assert [artifact["artifact"] for artifact in artifacts] == [
        "result",
        "source_experiment",
        "base_prompt",
    ]
    assert [artifact["path"] for artifact in artifacts] == [
        str(result_path),
        str(experiment_path),
        str(prompt_path),
    ]
    assert all(artifact["exists"] for artifact in artifacts)
    assert all(artifact["bytes"] > 0 for artifact in artifacts)
    assert all(artifact["modified_at"] for artifact in artifacts)


def test_load_result_files_supports_multiple_yaml_files(tmp_path):
    first = write_yaml(tmp_path, "first.yaml", sample_result_data())
    second = write_yaml(tmp_path, "second.yaml", sample_result_data())

    sources = load_result_files([first, second])

    assert [source.path for source in sources] == [first.resolve(), second.resolve()]


def test_html_payload_keeps_provider_filter_data_separate_from_provider_model(
    tmp_path,
):
    data = sample_result_data()
    data["metadata"]["llm_configurations"]["canary"]["phases"]["review"] = {
        "provider": "ollama",
        "model_identifier": "gemma3:4b",
    }
    source = load_result_file(write_yaml(tmp_path, "results.yaml", data))

    payload = _html_payload([source])

    overview = payload["overview"][0]
    question = payload["questions"][0]
    provider_rows = payload["provider_models"]

    assert overview["providers"] == ["openrouter", "ollama"]
    assert overview["provider"] == "openrouter, ollama"
    assert question["providers"] == ["openrouter", "ollama"]
    assert "main: openrouter/openai/gpt-test" in question["provider_model"]
    assert "review: ollama/gemma3:4b" in question["provider_model"]
    assert (
        provider_rows[0]["configuration_provider_model"] == question["provider_model"]
    )
