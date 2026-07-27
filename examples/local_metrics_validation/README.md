# External Local Metrics Validation

This directory contains validation-only tooling for checking Ollama local
resource metrics from outside the Heracles pipeline. It is intentionally kept
outside `src/heracles_agents/` so it is not part of the production metrics path.

## Example

From the `heracles_agents` repository root:

```bash
. .venv/bin/activate
python examples/local_metrics_validation/external_monitor.py \
  --experiment examples/experiments/ollama/canary_local_metrics_experiment.yaml \
  --output-dir output/local_metrics_validation/canary \
  --pre-pipeline-seconds 10 \
  --post-pipeline-seconds 10 \
  --pipeline-output-dir output \
  -- \
  python examples/experiment_runner.py \
    examples/experiments/ollama/canary_local_metrics_experiment.yaml \
    --debug \
    --no-display
```

The script starts external measurements, waits for `--pre-pipeline-seconds`,
runs the command after `--`, waits for `--post-pipeline-seconds`, then stops
measurements and writes artifacts under `--output-dir`.

## Outputs

- `external_local_metrics_samples.yaml`: raw samples, summary, preflight results,
  command metadata, and timing windows.
- `external_local_metrics_summary.yaml`: compact summary comparable to the
  in-pipeline local metrics shape.
- `external_timeseries.csv`: flattened samples for independent analysis.
- `event_timeline.yaml`: vertical plot marker positions for baseline,
  measurement, pipeline, and per-configuration events.
- `plots/`: resource usage plots over time.
  - `cpu_usage.png`
  - `memory_ram.png`
  - `gpu_usage.png`
  - `gpu_memory_vram.png`
  - `gpu_power.png`
  - `polling_telemetry.png`

## Baseline Timing

- The external monitor collects its GPU baseline before external runtime
  sampling starts. If `unload_models_before_baseline` is enabled, it first asks
  Ollama to unload currently loaded models, then samples GPU telemetry for
  `gpu_baseline_seconds`. The external resource plots use raw GPU telemetry;
  they do not subtract this baseline from runtime samples.
- The in-pipeline local metrics path does the same at the beginning of each
  experiment configuration. `experiment_runner.py` invokes one configuration at
  a time; each pipeline immediately calls `prepare_local_resource_monitor()`,
  which calls `LocalResourceMonitor.collect_baseline()` before the configuration
  resource measurement starts.
- For model sweeps, the model left loaded by one configuration is unloaded at
  the beginning of the next configuration's baseline step. The delay between
  configurations is therefore the unload request plus the configured
  `gpu_baseline_seconds` sampling window.
- There is no separate post-unload settling delay today; baseline sampling
  starts immediately after the unload request returns.
- Per-configuration baseline plot markers are inferred from
  `baseline_gpu_samples` in internal debug files. Those samples mark the baseline
  telemetry window, not the exact start/end timestamp of the Ollama unload API
  call.

## Notes

- Plots include vertical markers for external GPU baseline/model-unload start,
  external GPU baseline completion, external measurement start/stop, pipeline
  start/finish, and per-configuration baseline/measurement start/finish when
  internal debug samples are available.
- External metrics include the configured pre/post windows and the whole
  subprocess lifetime.
- Internal pipeline metrics only cover the pipeline's configured measurement
  window, so duration differences are expected.
- Docker CPU percentages need at least two samples.
- CPU usage plots show only normalized container CPU percentage, scaled to the
  0-100% range.
- Docker RAM plots use container process RSS from Docker `top` when available.
  This is a better approximation for model-resident RAM because Ollama/llama.cpp
  can hold model weights through memory-mapped files that Docker cgroup
  working-set accounting may classify as inactive file cache. Cgroup working
  set, raw usage including cache, file cache, inactive file, and anonymous memory
  are retained in the CSV and raw YAML for diagnosis.
- Ollama response runtime and token metrics cannot be independently sampled by
  this external monitor; compare those from pipeline result/debug files.

## Tests

```bash
. .venv/bin/activate
python -m pytest examples/local_metrics_validation/test_external_monitor.py -q
```
