from __future__ import annotations

import importlib.util
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from heracles_agents.local_metrics.docker_proxy import DockerContainerStatsSample
from heracles_agents.local_metrics.gpu_stats import GpuStatsSample


MODULE_PATH = Path(__file__).with_name("external_monitor.py")
SPEC = importlib.util.spec_from_file_location("external_monitor", MODULE_PATH)
external_monitor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(external_monitor)


def test_cli_parses_command_and_pre_post_delays():
    args = external_monitor.parse_args(
        [
            "--output-dir",
            "out",
            "--pre-pipeline-seconds",
            "1.5",
            "--post-pipeline-seconds",
            "2.5",
            "--",
            "python",
            "examples/experiment_runner.py",
            "--debug",
        ]
    )

    assert args.output_dir == Path("out")
    assert args.pre_pipeline_seconds == 1.5
    assert args.post_pipeline_seconds == 2.5
    assert args.command == ["python", "examples/experiment_runner.py", "--debug"]


def test_experiment_config_loads_and_cli_overrides_win(tmp_path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "metadata": {
                    "local_metrics": {
                        "enabled": True,
                        "ollama_container_name": "from-yaml",
                        "docker_host": "tcp://yaml-docker:2375",
                        "ollama_host": "http://yaml-ollama:11434",
                        "sample_interval_seconds": 0.5,
                        "gpu_baseline_seconds": 7,
                        "baseline_adjust_gpu": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = external_monitor.parse_args(
        [
            "--experiment",
            str(experiment),
            "--output-dir",
            str(tmp_path / "out"),
            "--sample-interval-seconds",
            "0.1",
            "--gpu-baseline-seconds",
            "2",
            "--ollama-container-name",
            "from-cli",
            "--docker-host",
            "tcp://cli-docker:2375",
            "--ollama-host",
            "http://cli-ollama:11434",
            "--no-baseline-adjust-gpu",
            "--",
            "true",
        ]
    )

    config = external_monitor.build_config(args)

    assert config.enabled is True
    assert config.ollama_container_name == "from-cli"
    assert config.docker_host == "tcp://cli-docker:2375"
    assert config.ollama_host == "http://cli-ollama:11434"
    assert config.sample_interval_seconds == 0.1
    assert config.gpu_baseline_seconds == 2
    assert config.baseline_adjust_gpu is False


def test_external_config_disables_gpu_baseline_adjustment_even_when_yaml_enables_it(
    tmp_path,
):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "metadata": {
                    "local_metrics": {
                        "enabled": True,
                        "baseline_adjust_gpu": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = external_monitor.parse_args(
        [
            "--experiment",
            str(experiment),
            "--output-dir",
            str(tmp_path / "out"),
            "--",
            "true",
        ]
    )

    assert external_monitor.build_config(args).baseline_adjust_gpu is False


def test_run_validation_stops_measurement_and_orders_pre_post_sleeps(tmp_path, monkeypatch):
    events = []

    class FakeMeasurement:
        def stop(self):
            events.append("stop")
            return _summary()

        def raw_samples(self):
            return _raw_samples()

    class FakeMonitor:
        def __init__(self, config):
            self.config = config

        def collect_baseline(self):
            events.append("baseline")

        def start(self):
            events.append("start")
            return FakeMeasurement()

    class FakeProcess:
        def wait(self):
            events.append("wait")
            return 42

    def fake_popen(command):
        events.append(("popen", command))
        return FakeProcess()

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    monkeypatch.setattr(external_monitor, "generate_plots", lambda *_args: None)
    args = external_monitor.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--pre-pipeline-seconds",
            "1.25",
            "--post-pipeline-seconds",
            "2.5",
            "--skip-preflight",
            "--",
            "python",
            "-c",
            "raise SystemExit(42)",
        ]
    )

    return_code = external_monitor.run_validation(
        args,
        sleep_fn=fake_sleep,
        popen_factory=fake_popen,
        monitor_factory=FakeMonitor,
        monotonic_fn=_monotonic_counter(),
        utcnow_fn=_utc_counter(),
    )

    assert return_code == 42
    assert events == [
        "baseline",
        "start",
        ("sleep", 1.25),
        ("popen", ["python", "-c", "raise SystemExit(42)"]),
        "wait",
        ("sleep", 2.5),
        "stop",
    ]
    summary = yaml.safe_load(
        (tmp_path / "external_local_metrics_summary.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert summary["pipeline_return_code"] == 42
    assert summary["validation_window"]["pre_pipeline_seconds"] == 1.25
    assert summary["validation_window"]["post_pipeline_seconds"] == 2.5
    assert not (tmp_path / "comparison_summary.yaml").exists()


def test_synthetic_docker_samples_produce_expected_cpu_timeseries():
    raw = {
        "docker_samples": [
            asdict(
                DockerContainerStatsSample(
                    timestamp_monotonic=10,
                    cpu_total_usage=100,
                    system_cpu_usage=1000,
                    online_cpus=4,
                    memory_usage_bytes=10,
                    memory_usage_including_cache_bytes=100,
                    memory_file_bytes=90,
                    memory_inactive_file_bytes=90,
                    memory_anon_bytes=10,
                    memory_limit_bytes=100,
                    pids_current=1,
                )
            ),
            asdict(
                DockerContainerStatsSample(
                    timestamp_monotonic=11,
                    cpu_total_usage=300,
                    system_cpu_usage=2000,
                    online_cpus=4,
                    memory_usage_bytes=20,
                    memory_usage_including_cache_bytes=200,
                    memory_file_bytes=180,
                    memory_inactive_file_bytes=180,
                    memory_anon_bytes=20,
                    memory_limit_bytes=100,
                    pids_current=2,
                )
            ),
        ],
        "baseline_gpu_samples": [],
        "gpu_samples": [],
    }

    rows = external_monitor.build_timeseries_rows(
        raw,
        sample_interval_seconds=1.0,
        baseline_adjust_gpu=True,
        measurement_start_monotonic=10,
    )

    assert rows[0]["container_cpu_percent"] is None
    assert rows[1]["container_cpu_percent"] == 80.0
    assert rows[1]["container_cpu_percent_normalized"] == 20.0
    assert rows[1]["container_ram_bytes"] == 20


def test_ram_plot_value_converts_bytes_to_mib():
    assert (
        external_monitor._plot_value(
            {"container_ram_bytes": 2 * 1024 * 1024},
            "container_ram_mib",
        )
        == 2.0
    )


def test_synthetic_gpu_samples_apply_baseline_adjustment():
    raw = {
        "docker_samples": [],
        "baseline_gpu_samples": [
            asdict(
                GpuStatsSample(
                    timestamp_monotonic=1,
                    gpu_index=0,
                    name="GPU",
                    memory_used_mib=100,
                    memory_total_mib=1000,
                    utilization_gpu_percent=5,
                    utilization_memory_percent=1,
                    power_draw_w=20,
                )
            )
        ],
        "gpu_samples": [
            asdict(
                GpuStatsSample(
                    timestamp_monotonic=10,
                    gpu_index=0,
                    name="GPU",
                    memory_used_mib=250,
                    memory_total_mib=1000,
                    utilization_gpu_percent=40,
                    utilization_memory_percent=15,
                    power_draw_w=70,
                )
            )
        ],
    }

    rows = external_monitor.build_timeseries_rows(
        raw,
        sample_interval_seconds=0.5,
        baseline_adjust_gpu=True,
        measurement_start_monotonic=10,
    )
    gpu_row = next(row for row in rows if row["sample_kind"] == "gpu")

    assert gpu_row["gpu_memory_used_mib_adjusted"] == 150
    assert gpu_row["gpu_utilization_percent_adjusted"] == 35
    assert gpu_row["gpu_power_w_adjusted"] == 50
    assert gpu_row["gpu_energy_wh_adjusted_cumulative"] == round(50 * 0.5 / 3600, 9)


def test_plot_generation_creates_nonempty_files(tmp_path):
    raw = _raw_samples()
    rows = external_monitor.build_timeseries_rows(
        raw,
        sample_interval_seconds=0.5,
        baseline_adjust_gpu=True,
        measurement_start_monotonic=10,
    )

    external_monitor.generate_plots(rows, tmp_path, "png", raw)

    expected = [
        "cpu_usage.png",
        "memory_ram.png",
        "gpu_usage.png",
        "gpu_memory_vram.png",
        "gpu_power.png",
        "polling_telemetry.png",
    ]
    for name in expected:
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_event_timeline_includes_pipeline_baseline_and_configuration_events(tmp_path):
    internal_path = tmp_path / "config_a_local_metrics_samples.yaml"
    internal_path.write_text(
        yaml.safe_dump(
            {
                "baseline_gpu_samples": [
                    {
                        "timestamp_monotonic": 122.0,
                        "gpu_index": 0,
                        "name": "GPU",
                        "memory_used_mib": 10,
                        "memory_total_mib": 100,
                        "utilization_gpu_percent": 1,
                        "utilization_memory_percent": 1,
                        "power_draw_w": 10,
                    },
                    {
                        "timestamp_monotonic": 128.0,
                        "gpu_index": 0,
                        "name": "GPU",
                        "memory_used_mib": 10,
                        "memory_total_mib": 100,
                        "utilization_gpu_percent": 1,
                        "utilization_memory_percent": 1,
                        "power_draw_w": 10,
                    },
                ],
                "docker_samples": [
                    {
                        "timestamp_monotonic": 130.0,
                        "cpu_total_usage": 1,
                        "system_cpu_usage": 1,
                        "online_cpus": 1,
                        "memory_usage_bytes": 1,
                        "memory_limit_bytes": 1,
                        "pids_current": 1,
                    },
                    {
                        "timestamp_monotonic": 140.0,
                        "cpu_total_usage": 2,
                        "system_cpu_usage": 2,
                        "online_cpus": 1,
                        "memory_usage_bytes": 1,
                        "memory_limit_bytes": 1,
                        "pids_current": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    events = external_monitor.build_event_timeline(
        {
            "baseline_started_monotonic": 90.0,
            "baseline_finished_monotonic": 99.0,
            "external_measurement_started_monotonic": 100.0,
            "pipeline_command_started_monotonic": 120.0,
            "pipeline_command_finished_monotonic": 150.0,
            "external_measurement_stopped_monotonic": 160.0,
        },
        {},
        [internal_path],
        unload_models_before_baseline=True,
    )

    by_name = {event["name"]: event for event in events}
    assert by_name["baseline_started"]["relative_seconds"] == -10.0
    assert by_name["pipeline_started"]["relative_seconds"] == 20.0
    assert by_name["pipeline_finished"]["relative_seconds"] == 50.0
    assert by_name["configuration_baseline_started:config_a"]["relative_seconds"] == 22.0
    assert by_name["configuration_baseline_finished:config_a"]["relative_seconds"] == 28.0
    assert by_name["configuration_started:config_a"]["relative_seconds"] == 30.0
    assert by_name["configuration_finished:config_a"]["relative_seconds"] == 40.0


def _raw_samples():
    return {
        "poll_durations_seconds": {
            "docker": [0.01, 0.02],
            "gpu": [0.03],
        },
        "docker_samples": [
            asdict(
                DockerContainerStatsSample(
                    timestamp_monotonic=10,
                    cpu_total_usage=100,
                    system_cpu_usage=1000,
                    online_cpus=4,
                    memory_usage_bytes=10,
                    memory_usage_including_cache_bytes=100,
                    memory_file_bytes=90,
                    memory_inactive_file_bytes=90,
                    memory_anon_bytes=10,
                    memory_limit_bytes=100,
                    pids_current=1,
                )
            ),
            asdict(
                DockerContainerStatsSample(
                    timestamp_monotonic=11,
                    cpu_total_usage=300,
                    system_cpu_usage=2000,
                    online_cpus=4,
                    memory_usage_bytes=20,
                    memory_usage_including_cache_bytes=200,
                    memory_file_bytes=180,
                    memory_inactive_file_bytes=180,
                    memory_anon_bytes=20,
                    memory_limit_bytes=100,
                    pids_current=2,
                )
            ),
        ],
        "baseline_gpu_samples": [
            asdict(
                GpuStatsSample(
                    timestamp_monotonic=1,
                    gpu_index=0,
                    name="GPU",
                    memory_used_mib=100,
                    memory_total_mib=1000,
                    utilization_gpu_percent=5,
                    utilization_memory_percent=1,
                    power_draw_w=20,
                )
            )
        ],
        "gpu_samples": [
            asdict(
                GpuStatsSample(
                    timestamp_monotonic=10,
                    gpu_index=0,
                    name="GPU",
                    memory_used_mib=250,
                    memory_total_mib=1000,
                    utilization_gpu_percent=40,
                    utilization_memory_percent=15,
                    power_draw_w=70,
                )
            )
        ],
    }


def _summary():
    return {
        "measurement_scope": "ollama_container",
        "ollama_container_name": "ollama",
        "sample_interval_seconds": 0.025,
        "baseline_seconds": 5,
        "baseline_adjusted": True,
        "telemetry": {
            "measurement_duration_seconds": 10,
            "docker_effective_interval_seconds_avg": 0.1,
        },
        "cpu": {"container_cpu_percent_avg": 50, "sample_count": 2},
        "ram": {"container_ram_bytes_avg": 15, "sample_count": 2},
        "gpu": {"gpu_power_w_adjusted_avg": 50, "gpu_sample_count": 1},
        "warnings": [],
    }


def _monotonic_counter():
    value = {"current": 100.0}

    def current():
        value["current"] += 1.0
        return value["current"]

    return current


def _utc_counter():
    value = {"current": datetime(2026, 7, 27, tzinfo=UTC)}

    def current():
        value["current"] += timedelta(seconds=1)
        return value["current"]

    return current
