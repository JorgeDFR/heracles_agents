from types import SimpleNamespace
from unittest.mock import patch

import pytest

from heracles_agents.llm_interface import AgentContext
from heracles_agents.local_metrics.docker_proxy import (
    DockerContainerStatsSample,
    DockerProxyClient,
    calculate_cpu_percent,
    docker_host_to_http_base_url,
    docker_stats_sample_from_dict,
    summarize_docker_samples,
)
from heracles_agents.local_metrics.gpu_stats import (
    GpuProcessSample,
    GpuStatsSample,
    parse_gpu_processes,
    parse_gpu_stats,
    summarize_gpu_samples,
)
from heracles_agents.local_metrics.monitor import (
    LocalMetricsConfig,
    LocalResourceMonitor,
)
from heracles_agents.local_metrics.ollama_runtime import (
    extract_ollama_response_metrics,
    summarize_ollama_metrics,
)


def test_docker_proxy_url_parser_supports_tcp_and_http():
    assert (
        docker_host_to_http_base_url("tcp://docker-socket-proxy:2375")
        == "http://docker-socket-proxy:2375"
    )
    assert (
        docker_host_to_http_base_url("http://docker-socket-proxy:2375/")
        == "http://docker-socket-proxy:2375"
    )

    with pytest.raises(ValueError, match="Unsupported Docker host"):
        docker_host_to_http_base_url("unix:///var/run/docker.sock")


def test_docker_stats_parser_and_cpu_percent():
    previous = DockerContainerStatsSample(
        timestamp_monotonic=1.0,
        cpu_total_usage=100,
        system_cpu_usage=1000,
        online_cpus=4,
        memory_usage_bytes=10,
        memory_limit_bytes=100,
        pids_current=2,
    )
    current = DockerContainerStatsSample(
        timestamp_monotonic=2.0,
        cpu_total_usage=300,
        system_cpu_usage=2000,
        online_cpus=4,
        memory_usage_bytes=30,
        memory_limit_bytes=100,
        pids_current=5,
    )

    assert calculate_cpu_percent(previous, current) == 80.0
    assert calculate_cpu_percent(current, previous) is None
    cpu, ram = summarize_docker_samples([previous, current])
    assert cpu["container_cpu_percent_avg"] == 80.0
    assert cpu["container_cpu_percent_peak"] == 80.0
    assert cpu["container_cpu_percent_normalized_avg"] == 20.0
    assert cpu["container_cpu_percent_normalized_peak"] == 20.0
    assert cpu["container_online_cpus"] == 4
    assert ram["container_ram_bytes_avg"] == 20
    assert ram["container_ram_bytes_peak"] == 30
    assert ram["container_ram_limit_bytes"] == 100
    assert ram["container_pids_peak"] == 5


def test_docker_stats_sample_from_api_dict():
    sample = docker_stats_sample_from_dict(
        {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 10},
                "system_cpu_usage": 20,
                "online_cpus": 8,
            },
            "memory_stats": {"usage": 30, "limit": 40},
            "pids_stats": {"current": 3},
        }
    )

    assert sample.cpu_total_usage == 10
    assert sample.system_cpu_usage == 20
    assert sample.online_cpus == 8
    assert sample.memory_usage_bytes == 30
    assert sample.memory_limit_bytes == 40
    assert sample.pids_current == 3


def test_docker_proxy_client_parses_container_stats_response():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'{"cpu_stats":{"cpu_usage":{"total_usage":10},'
                b'"system_cpu_usage":20,"online_cpus":2},'
                b'"memory_stats":{"usage":30,"limit":40},'
                b'"pids_stats":{"current":5}}'
            )

    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        sample = DockerProxyClient("tcp://docker-socket-proxy:2375").container_stats(
            "ollama"
        )

    assert sample.memory_usage_bytes == 30
    assert (
        "containers/ollama/stats?stream=false&one-shot=true"
        in urlopen.call_args.args[0].full_url
    )


def test_gpu_csv_parsers_and_baseline_adjustment():
    gpu_samples = parse_gpu_stats(
        "0, NVIDIA GPU, 1000, 16000, 50, 20, 120.5\n"
        "1, Other GPU, 200, 8000, 0, 0, N/A\n"
    )
    processes = parse_gpu_processes("123, /bin/ollama, 900\n")

    assert gpu_samples[0].memory_used_mib == 1000
    assert gpu_samples[0].power_draw_w == 120.5
    assert gpu_samples[1].power_draw_w is None
    assert processes[0].pid == 123
    assert processes[0].used_gpu_memory_mib == 900

    summary = summarize_gpu_samples(
        [gpu_samples[0]],
        [
            GpuStatsSample(
                timestamp_monotonic=0.0,
                gpu_index=0,
                name="NVIDIA GPU",
                memory_used_mib=400,
                memory_total_mib=16000,
                utilization_gpu_percent=5,
                utilization_memory_percent=0,
                power_draw_w=20,
            )
        ],
        sample_interval_seconds=0.5,
        process_samples=processes,
    )

    assert summary["gpu_memory_used_mib_adjusted_peak"] == 600
    assert summary["gpu_utilization_percent_adjusted_avg"] == 45
    assert summary["gpu_power_w_adjusted_avg"] == 100.5
    assert summary["gpu_energy_wh_adjusted"] == round(100.5 * 0.5 / 3600.0, 9)
    assert summary["ollama_process_vram_available"] is True
    assert summary["ollama_process_vram_mib_peak"] == 900


def test_ollama_response_metric_extraction_and_summary():
    response = SimpleNamespace(
        total_duration=10_000_000_000,
        load_duration=1_000_000_000,
        prompt_eval_count=100,
        prompt_eval_duration=2_000_000_000,
        eval_count=40,
        eval_duration=4_000_000_000,
    )

    metrics = extract_ollama_response_metrics(response)

    assert metrics["total_duration_seconds"] == 10
    assert metrics["load_duration_seconds"] == 1
    assert metrics["prompt_tokens_per_second"] == 50
    assert metrics["output_tokens_per_second"] == 10
    assert metrics["steady_state_seconds"] == 6
    summary = summarize_ollama_metrics([metrics, metrics])
    assert summary["eval_count"] == 80
    assert summary["output_tokens_per_second"] == 10


def test_agent_context_records_ollama_runtime_metrics():
    response = SimpleNamespace(
        total_duration=1_000_000_000,
        load_duration=0,
        prompt_eval_count=2,
        prompt_eval_duration=1_000_000_000,
        eval_count=4,
        eval_duration=1_000_000_000,
    )
    agent = SimpleNamespace(client=SimpleNamespace(client_type="ollama"))
    context = AgentContext(agent)

    context.record_local_llm_runtime_metrics(response)

    assert context.local_llm_runtime_metrics[0]["output_tokens_per_second"] == 4


def test_monitor_returns_warnings_when_sources_are_unavailable():
    monitor = LocalResourceMonitor(
        LocalMetricsConfig(
            enabled=True,
            docker_host="bad://docker",
            gpu_baseline_seconds=0,
            sample_interval_seconds=0.01,
        )
    )

    with (
        patch(
            "heracles_agents.local_metrics.monitor.sample_gpu_stats",
            side_effect=RuntimeError("no gpu"),
        ),
        patch(
            "heracles_agents.local_metrics.monitor.sample_gpu_processes",
            side_effect=RuntimeError("no processes"),
        ),
    ):
        measurement = monitor.start()
        measurement._sample_once()
        summary = measurement.stop()

    assert summary["cpu"]["sample_count"] == 0
    assert any("Docker proxy metrics unavailable" in w for w in summary["warnings"])
    assert any("nvidia-smi GPU metrics unavailable" in w for w in summary["warnings"])
