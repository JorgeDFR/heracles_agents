from types import SimpleNamespace
from unittest.mock import patch

import pytest

from heracles_agents.llm_interface import AgentContext
from heracles_agents.local_metrics.docker_proxy import (
    DockerContainerStatsSample,
    DockerProxyClient,
    apply_process_memory_from_top,
    calculate_cpu_percent,
    docker_host_to_http_base_url,
    docker_stats_sample_from_dict,
    process_memory_from_top,
    summarize_docker_samples,
)
from heracles_agents.local_metrics.gpu_stats import (
    GpuStatsSample,
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
from heracles_agents.pipelines.local_metrics import _warm_up_ollama


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
        memory_cgroup_working_set_bytes=10,
        memory_limit_bytes=100,
        pids_current=2,
    )
    current = DockerContainerStatsSample(
        timestamp_monotonic=2.0,
        cpu_total_usage=300,
        system_cpu_usage=2000,
        online_cpus=4,
        memory_usage_bytes=30,
        memory_cgroup_working_set_bytes=30,
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
    assert ram["container_ram_measurement_source"] == "docker_cgroup_working_set"
    assert ram["container_process_rss_available"] is False
    assert ram["container_ram_working_set_bytes_avg"] == 20
    assert ram["container_ram_working_set_bytes_peak"] == 30
    assert ram["container_ram_limit_bytes"] == 100
    assert ram["container_pids_peak"] == 5


def test_docker_stats_sample_from_api_dict_uses_working_set_memory():
    sample = docker_stats_sample_from_dict(
        {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 10},
                "system_cpu_usage": 20,
                "online_cpus": 8,
            },
            "memory_stats": {
                "usage": 30,
                "limit": 40,
                "stats": {
                    "inactive_file": 8,
                    "file": 20,
                    "active_file": 12,
                    "anon": 6,
                },
            },
            "pids_stats": {"current": 3},
        }
    )

    assert sample.cpu_total_usage == 10
    assert sample.system_cpu_usage == 20
    assert sample.online_cpus == 8
    assert sample.memory_usage_bytes == 22
    assert sample.memory_cgroup_working_set_bytes == 22
    assert sample.memory_usage_including_cache_bytes == 30
    assert sample.memory_file_bytes == 20
    assert sample.memory_active_file_bytes == 12
    assert sample.memory_inactive_file_bytes == 8
    assert sample.memory_anon_bytes == 6
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
                b'"memory_stats":{"usage":30,"limit":40,'
                b'"stats":{"inactive_file":10,"file":20,"anon":5}},'
                b'"pids_stats":{"current":5}}'
            )

    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        sample = DockerProxyClient("tcp://docker-socket-proxy:2375").container_stats(
            "ollama"
        )

    assert sample.memory_usage_bytes == 20
    assert sample.memory_usage_including_cache_bytes == 30
    assert (
        "containers/ollama/stats?stream=false&one-shot=true"
        in urlopen.call_args_list[0].args[0].full_url
    )


def test_docker_top_process_memory_replaces_default_ram_estimate():
    sample = DockerContainerStatsSample(
        timestamp_monotonic=1,
        cpu_total_usage=None,
        system_cpu_usage=None,
        online_cpus=None,
        memory_usage_bytes=100,
        memory_cgroup_working_set_bytes=100,
        memory_limit_bytes=1000,
        pids_current=2,
    )
    top = {
        "Titles": ["PID", "PPID", "RSS", "VSZ", "COMMAND", "COMMAND"],
        "Processes": [
            ["10", "1", "200", "1000", "ollama", "/bin/ollama serve"],
            ["11", "10", "300", "2000", "llama-server", "llama-server"],
        ],
    }

    parsed = process_memory_from_top(top)
    apply_process_memory_from_top(sample, top)
    _, ram = summarize_docker_samples([sample])

    assert parsed["process_rss_bytes"] == 500 * 1024
    assert parsed["process_vsz_bytes"] == 3000 * 1024
    assert sample.memory_usage_bytes == 500 * 1024
    assert sample.memory_cgroup_working_set_bytes == 100
    assert ram["container_ram_bytes_peak"] == 500 * 1024
    assert ram["container_ram_measurement_source"] == "docker_top_process_rss"
    assert ram["container_process_rss_available"] is True
    assert ram["container_ram_working_set_bytes_peak"] == 100
    assert ram["container_process_rss_bytes_peak"] == 500 * 1024


def test_gpu_csv_parsers_and_baseline_adjustment():
    gpu_samples = parse_gpu_stats(
        "0, NVIDIA GPU, 1000, 16000, 50, 20, 120.5\n"
        "1, Other GPU, 200, 8000, 0, 0, N/A\n"
    )

    assert gpu_samples[0].memory_used_mib == 1000
    assert gpu_samples[0].power_draw_w == 120.5
    assert gpu_samples[1].power_draw_w is None

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
    )

    assert summary["gpu_memory_used_mib_adjusted_peak"] == 600
    assert summary["gpu_utilization_percent_adjusted_avg"] == 45
    assert summary["gpu_power_w_adjusted_avg"] == 100.5
    assert summary["gpu_energy_wh_adjusted"] == round(100.5 * 0.5 / 3600.0, 9)


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


def test_ollama_warmup_runs_before_measurement_and_records_result(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"{}"

    requests = []
    monkeypatch.setattr(
        "heracles_agents.pipelines.local_metrics.urllib.request.urlopen",
        lambda request, timeout: requests.append((request, timeout)) or Response(),
    )
    monitor = LocalResourceMonitor(
        LocalMetricsConfig(enabled=True, warmup_enabled=True, warmup_requests=1)
    )
    exp = SimpleNamespace(
        phases={
            "main": SimpleNamespace(
                client=SimpleNamespace(client_type="ollama"),
                model_info=SimpleNamespace(model="test-model"),
            )
        }
    )

    _warm_up_ollama(exp, monitor)

    assert len(requests) == 1
    assert monitor.warmup["requests"][0]["model"] == "test-model"
    assert monitor.warmup["requests"][0]["succeeded"] is True


def test_monitor_returns_warnings_when_sources_are_unavailable():
    monitor = LocalResourceMonitor(
        LocalMetricsConfig(
            enabled=True,
            docker_host="bad://docker",
            gpu_baseline_seconds=0,
            sample_interval_seconds=0.01,
        )
    )

    with patch(
        "heracles_agents.local_metrics.monitor.sample_gpu_stats",
        side_effect=RuntimeError("no gpu"),
    ):
        measurement = monitor.start()
        measurement._sample_once()
        summary = measurement.stop()

    assert summary["cpu"]["sample_count"] == 0
    assert any("Docker proxy metrics unavailable" in w for w in summary["warnings"])
    assert any("nvidia-smi GPU metrics unavailable" in w for w in summary["warnings"])
