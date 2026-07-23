# Heracles Agents

>**Note:** This is a fork of the upstream Heracles Agents framework. This fork preserves the original agent architecture while adding experimental tooling, providers, and experiments for robotics and spatial reasoning research.

`heracles_agents` is a minimal agentic LLM framework. It has been developed
with a focus on the following priorities:

* Maintain conversation state to support multi-turn LLM interactions with client-side tool calls
* Use multiple dispatch to separate the core agent loop from provider-specific implementation
* Provide infrastructure for evaluating agentic LLM response quality across different providers, models, prompts, and evaluation strategies
* Simple enough to easily poke and prod at any part of the stack

The `heracles_agents` library should be flexible enough to be directly
integrated in other downstream applications. In this repository we provide two
example uses:
* An interactive agent that allows the user to query and update a 3D scene graph and send goals to robots, and
* An experimental pipeline built to evaluate agentic LLM responses for different providers, models, prompts, and evaluation strategies.
See the Examples section below for more detail on these programs.

![Chat DSG](img/chatdsg_screenshot.png "ChatDSG example interaction")

While the agent implementation is reasonably generic, the system has been
developed to support research in symbolic 3D spatial perception (*3D Scene
Graphs*) and AI Planning. To that end, we provide LLM tools for
* [Executing Cypher queries against a graph database](src/heracles_agents/tools/cypher_query_tool.py)
* [Sending PDDL goals to robots](src/heracles_agents/tools/pddl_calling_tool.py)
* [Sending waypoints to quadrotors](src/heracles_agents/tools/penn_integration_tool.py)
* [Highlighting objects of interest in a 3D scene graph](src/heracles_agents/tools/visualize_objects_tool.py)



## Installation

Note that this repo requires Python 3.12 and has only been tested on Ubuntu
24.04. It should work on Ubuntu 22.04, as long as you don't use the ROS
functionality.

While `heracles_agents` can be used as a standalone minimal agent framework,
you probably want to also install `heracles`, which provides an interface
between 3D scene graphs and a graph database.

First, we'll install `heracles`:
```bash
git clone git@github.com:GoldenZephyr/heracles.git
pip install ./heracles/heracles
```

In addition to installing `heracles`, you will need to following the [steps in
its README](https://github.com/GoldenZephyr/heracles) for running the graph
database and installing `spark_dsg` if you would like to use the scene graph
functionality.


Next, instead `heracles_agents`:
```bash
git clone git@github.com:GoldenZephyr/heracles_agents.git
pip install heracles_agents
```

### Heracles ROS

If you intend to use Heracles as part of a ROS-based system, you can put
`heracles` and `heracles_agents` in your ROS workspace instead of
pip-installing them manually, as long as your virtual environment was created
with the `--system-site-packages` option (which you would want anyway).

### Environment Variables

| Environment Variable Name         | Description                                                                |
|-----------------------------------|----------------------------------------------------------------------------|
| HERACLES\_OPENAI\_API\_KEY        | OpenAI API key to use                                                      |
| HERACLES\_ANTHROPIC\_API\_KEY     | Anthropic API key to use                                                   |
| AWS\_BEARER\_TOKEN\_BEDROCK       | If you want to use Bedrock                                                 |
| HERACLES\_OPENROUTER\_API\_KEY    | OpenRouter API key to use                                                  |
| OLLAMA\_HOST                      | Ollama server URL, e.g. http://ollama:11434                                |
| HUGGINGFACE\_HOST                 | Hugging Face-compatible model server URL, e.g. http://huggingface:8000     |
| HERACLES\_EVALUATION\_PATH        | Path to where this repo is cloned (only necessary for the example prompts  |
| HERACLES\_NEO4J\_USERNAME         | Username of local Neo4j graph database                                     |
| HERACLES\_NEO4J\_PASSWORD         | Password of local Neo4j graph database                                     |
| HERACLES\_NEO4J\_URI              | Address for database (neo4j://IP:PORT)                                     |
| ADT4\_HERACLES\_IP                | Same database IP as above (necessary for the LLM agent demo)               |
| ADT4\_HERACLES\_PORT              | Same database PORT as above (necessary for the LLM agent demo)             |
| HERACLES\_VENV                    | Path to your virtualenv directory (only necessary for full agent demo.)    |
| ROS2\_WS                          | Path to your ROS2 workspace (only necessary for full agent  demo.)         |

## Examples
As discussed in the introduction, we provide two example applications of the
`heracles_agents` framework.

### Chatdsg

[chatdsg.py](examples/chatdsg/chatdsg.py) is a simple terminal-based interface
to a tool-enabled LLM agent. The agent can answer questions or make updates to
a 3D scene graph if you have an instance of
[heracles](https://github.com/GoldenZephyr/heracles) running. You can also send
PDDL goals to an instance of
[Omniplanner](https://github.com/MIT-SPARK/Omniplanner), or highlight scene
graph objects of interest in Rviz. You can modify the LLM model or tools that
are used [in the config file](examples/chatdsg/agent_config.yaml), or [change the
prompt file](examples/chatdsg/agent_prompt.yaml).

If you want to actually visualize the 3D scene graph (and any edits you make to
it), you will need to install ROS2 and Hydra. Please see the [ROS2
documentation](https://docs.ros.org/en/jazzy/Installation.html) for
instructions to install ROS, and the [Hydra
documentation](https://github.com/MIT-SPARK/Hydra-ROS/) repository for
instructions to install Hydra. Once you have installed ROS2 and Hydra, you can
run the [chatdsg launch script](examples/chatdsg/chatdsg_system_example.yaml)
with the command:

```bash
tmuxp load chatdsg_system_example.yaml
```

Refer to the table at the top of this README for the environment variables that
must be set.

### Experiment Pipelines

The experiment runner can execute configured question sets against one or more
agent/prompt/model combinations and write structured results for later
comparison. Example configurations live under
[examples/experiments](examples/experiments), with provider-specific variants
for OpenAI, OpenRouter, Ollama, Anthropic, and Bedrock.

To run an experiment, install the package, configure the relevant provider API
key or local service, then pass one or more experiment YAML files to the runner:

```bash
python examples/experiment_runner.py examples/experiments/openai/cypher_experiment.yaml
```

The runner writes results to `output/<experiment-folder>/<experiment-name>_results.yaml`
by default.

Useful options:

```bash
# Run multiple experiment files
python examples/experiment_runner.py \
  examples/experiments/openai/canary_experiment.yaml \
  examples/experiments/openrouter/canary_experiment.yaml

# Run only one named configuration from an experiment file
python examples/experiment_runner.py \
  examples/experiments/tests/openai_test.yaml \
  --configuration agentic-canary-qa

# Choose an output directory and suppress live result tables
python examples/experiment_runner.py \
  examples/experiments/openai/cypher_experiment.yaml \
  --output-dir output \
  --no-display
```

OpenRouter and Ollama model sweeps can be configured with a reusable model list.
The OpenRouter provided example sweep uses
[examples/experiments/openrouter/model_lists/example.yaml](examples/experiments/openrouter/model_lists/example.yaml)
while the Ollama provided example sweep uses
[examples/experiments/ollama/model_lists/example.yaml](examples/experiments/ollama/model_lists/example.yaml).
Future model changes should usually only require editing those files.

```bash
# Test one model from the Cypher Ollama model sweep
python examples/experiment_runner.py \
  examples/experiments/ollama/cypher_model_sweep.yaml \
  --configuration agentic-cypher-qa-gemma4-12b \
  --output-dir output/model_sweep

# Run the full Cypher and PDDL OpenRouter model sweeps
python examples/experiment_runner.py \
  examples/experiments/openrouter/cypher_model_sweep.yaml \
  examples/experiments/openrouter/pddl_model_sweep.yaml \
  --output-dir output/model_sweep \
  --no-display
```

Model sweep runs write one result YAML per generated model configuration under
`output/<output-dir>/<openrouter|ollama>/<experiment-name>/`. The file name includes the
model alias, while the configuration name inside each result YAML stays stable
for grouping, such as `agentic-cypher-qa` or `agentic-cypher-pddl`.

To inspect saved result YAML files later, use the terminal summary:

```bash
python examples/display_yaml_results.py output/openrouter/cypher_experiment_results.yaml
```

The terminal view is intentionally limited to question text, solution, answer,
and core quality/token/tool metrics so it remains readable at normal terminal
widths.

For full metric analysis and cross-run comparison, generate a self-contained
HTML report:

```bash
python examples/display_yaml_results.py output/openrouter/*_results.yaml \
  --mode html \
  --output output/openrouter/report.html

python examples/display_yaml_results.py \
  output/model_sweep/openrouter/cypher_model_sweep/*_results.yaml \
  output/model_sweep/openrouter/pddl_model_sweep/*_results.yaml \
  output/model_sweep/ollama/cypher_model_sweep/*_results.yaml \
  output/model_sweep/ollama/pddl_model_sweep/*_results.yaml \
  --mode html \
  --output output/model_sweep/report.html
```

The HTML report includes all recorded latency metrics, sequence details, and
OpenRouter cost fields when those fields are present in the YAML.

The prompt templates used by those configurations are under
[examples/prompts](examples/prompts), and example question sets are under
[examples/questions](examples/questions).


## Custom Tools

Adding new tools for an LLM to use is quite straightforward. Currently, the
function metadata presented to the LLM is explicitly annotated external to the
function definition (as opposed to relying on inline annotation). We believe
this encourages reuse of existing functions and code that is easier to read.

Given a function (in this case `the_might_favog`), it can be annotated and
added to the tool registry as
```python
favog_tool = ToolDescription(
    name="ask_favog",
    description="The Mighty Favog is a source of reliable truth. Ask him anything you don't know. Please categorize your query as business, sports, or personal.",
    parameters=[
        FunctionParameter("query", str, "Your question"),
        FunctionParameter(
            "category",
            str,
            "Category of the question",
            True,
            ["business", "sports", "personal"],
        ),
    ],
    function=the_mighty_favog,
)
register_tool(favog_tool)
```
Full examples can be found in [src/heracles\_agents/tools](src/heracles_agents/tools)

## LLM Providers

Currently, `heracles_agents` supports the following LLM providers:
* openai
* anthropic
* bedrock
* openrouter
* ollama
* huggingface (custom HTTP server)

We implement a "hand rolled" tool call implementation (i.e., LLM's express
their intent to call a tool as part of the normal response body, as opposed to
a special tool response), in addition to provider-specific tool call
interfaces. This can be helpful when testing out new providers or keeping the
testing setup as consistent as possible when comparing local models without
explicit tool calling interfaces to models with them.

Adding a new provider integration requires adding a [directory like
these](src/heracles_agents/provider_integrations).

## Reference
If you use this library, please cite us with the following:
```bibtex
@misc{ray2025structuredinterfaces,
      title={Structured Interfaces for Automated Reasoning with {3D} Scene Graphs},
      author={Aaron Ray and Jacob Arkin and Harel Biggie and Chuchu Fan and Luca Carlone and Nicholas Roy},
      year={2025},
      eprint={2510.16643},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.16643},
}
```
