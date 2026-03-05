import logging
import os
import yaml

from heracles_agents.experiment_definition import ExperimentDescription
from heracles_agents.llm_interface import AnalyzedExperiment
from heracles_agents.summarize_results import display_experiment_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

experiment_fn = "experiments/canary_experiment.yaml"
# experiment_fn = "experiments/ollama_test.yaml"
# experiment_fn = "experiments/openai_test.yaml"
# experiment_fn = "experiments/anthropic_test.yaml"
# experiment_fn = "experiments/bedrock_test.yaml"

# Load experiment YAML
with open(experiment_fn, "r") as fo:
    yml = yaml.safe_load(fo)

experiment = ExperimentDescription(**yml)
logger.debug(f"Loaded experiment: {experiment}")

# Run experiment for each configuration
results = {}
for configuration_name, experiment_config in experiment.configurations.items():
    logger.info(f"Testing configuration: {configuration_name}")
    analyzed_questions = experiment_config.pipeline.function(experiment_config)

    display_experiment_results(analyzed_questions)
    results[configuration_name] = analyzed_questions

# Aggregate results
ae = AnalyzedExperiment(experiment_configurations=results)

# Prepare output folder and file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
os.makedirs(output_dir, exist_ok=True)

# Create a filename based on experiment name (sanitized)
experiment_name_safe = os.path.splitext(os.path.basename(experiment_fn))[0].replace(" ", "_").lower()
output_file = os.path.join(output_dir, f"{experiment_name_safe}_results.yaml")

# Write results to YAML
with open(output_file, "w") as fo:
    yaml.dump(ae.model_dump(), fo)

logger.info(f"Experiment results saved to {output_file}")