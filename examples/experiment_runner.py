import logging
import os
import yaml

from heracles_agents.experiment_definition import ExperimentDescription
from heracles_agents.llm_interface import AnalyzedExperiment
from heracles_agents.summarize_results import display_experiment_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN, force=True)

# List of experiment files
# experiment_fns = [
#     "experiments/ollama/canary_experiment.yaml",
#     "experiments/ollama/cypher_experiment.yaml",
#     "experiments/ollama/pddl_experiment.yaml",
#     "experiments/ollama/cypher_feedforward_experiment.yaml",
#     "experiments/ollama/pddl_feedforward_experiment.yaml",
# ]

experiment_fns = [
    "experiments/openai/canary_experiment.yaml",
    "experiments/openai/cypher_experiment.yaml",
    "experiments/openai/pddl_experiment.yaml",
    "experiments/openai/cypher_feedforward_experiment.yaml",
    "experiments/openai/pddl_feedforward_experiment.yaml",
]

# experiment_fns = [
#     "experiments/tests/ollama_test.yaml"
#     "experiments/tests/openai_test.yaml"
#     "experiments/tests/anthropic_test.yaml"
#     "experiments/tests/bedrock_test.yaml"
# ]

# Prepare base output folder
script_dir = os.path.dirname(os.path.abspath(__file__))
base_output_dir = os.path.join(script_dir, "..", "output")
os.makedirs(base_output_dir, exist_ok=True)

# Run all experiments
for experiment_fn in experiment_fns:
    logger.info(f"\n=== Running experiment: {experiment_fn} ===")

    # Load experiment YAML
    with open(experiment_fn, "r") as fo:
        yml = yaml.safe_load(fo)

    experiment = ExperimentDescription(**yml)
    logger.debug(f"Loaded experiment: {experiment}")

    # Run experiment configurations
    results = {}
    for configuration_name, experiment_config in experiment.configurations.items():
        logger.info(f"Testing configuration: {configuration_name}")
        analyzed_questions = experiment_config.pipeline.function(experiment_config)

        display_experiment_results(analyzed_questions)
        results[configuration_name] = analyzed_questions

    # Aggregate results
    ae = AnalyzedExperiment(experiment_configurations=results)

    # Create per-experiment output folder
    parent_folder = os.path.basename(os.path.dirname(experiment_fn))
    output_dir = os.path.join(base_output_dir, parent_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    experiment_name = os.path.splitext(os.path.basename(experiment_fn))[0] \
        .replace(" ", "_").lower()
    output_file = os.path.join(output_dir, f"{experiment_name}_results.yaml")

    # Write results
    with open(output_file, "w") as fo:
        yaml.dump(ae.model_dump(), fo)

    logger.info(f"Experiment results saved to {output_file}")