from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from heracles_agents.llm_agent import AgentInfo, ModelInfo, apply_bound_args
from heracles_agents.llm_interface import PddlComparison, SldpComparison
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.prompt import (
    InContextExample,
    Prompt,
    PromptSettings,
)
from heracles_agents.pipelines.prompt_utils import (
    get_pddl_answer_tag_text,
    get_pddl_format_description,
    get_sldp_answer_tag_text,
    get_sldp_format_description,
)
from heracles_agents.provider_integrations.anthropic.prompt_rendering import (
    render_anthropic_prompt,
)
from heracles_agents.provider_integrations.bedrock.prompt_rendering import (
    render_bedrock_example,
    render_bedrock_prompt,
)
from heracles_agents.provider_integrations.openai.prompt_rendering import (
    render_openai_example,
    render_openai_prompt,
)
from heracles_agents.provider_integrations.ollama.prompt_rendering import (
    render_ollama_prompt,
)
from heracles_agents.provider_integrations.openrouter.prompt_rendering import (
    render_openrouter_prompt,
)
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription
from heracles_agents.tool_calling.registry import ToolRegistry


class BoundConfig(BaseModel):
    prefix: str
    count: int


def test_model_info_normalizes_reasoning_settings():
    normalized = ModelInfo(
        model="model",
        temperature=None,
        seed=None,
        reasoning={"mode": "enabled", "effort": "custom-level"},
    )
    legacy = ModelInfo(model="model", reasoning="medium")

    assert normalized.reasoning.mode == "enabled"
    assert normalized.reasoning.effort == "custom-level"
    assert normalized.temperature is None
    assert legacy.reasoning.mode == "enabled"
    assert legacy.reasoning.effort == "medium"


def test_model_info_rejects_effort_when_reasoning_is_not_enabled():
    with pytest.raises(ValueError, match="only be set when reasoning mode is enabled"):
        ModelInfo(
            model="model",
            reasoning={"mode": "disabled", "effort": "low"},
        )


def test_evaluate_answer_handles_valid_and_invalid_pddl():
    comparator = PddlComparison(comparison_type="PDDL", relation="equal")

    assert evaluate_answer(comparator, "(and)", "(and)") == (True, True)
    assert evaluate_answer(comparator, "(and)", "(or)") == (True, False)
    assert evaluate_answer(comparator, "(", "(and)") == (False, False)


def test_evaluate_answer_handles_valid_and_invalid_sldp():
    comparator = SldpComparison(comparison_type="SLDP", relation="equal")

    assert evaluate_answer(comparator, "<1, 2>", "<2, 1>") == (True, True)
    assert evaluate_answer(comparator, "<1>", "<2>") == (True, False)
    assert evaluate_answer(comparator, "(", "<2>") == (False, False)


def test_apply_bound_args_instantiates_scalars_and_models():
    ToolRegistry.tools.clear()

    def configured_tool(config: BoundConfig, name: str, scale: float = 1.0):
        return f"{config.prefix}:{name}:{config.count * scale}"

    tool = ToolDescription(
        name="configured",
        description="Configured tool",
        parameters=[
            FunctionParameter("config", BoundConfig, "Config"),
            FunctionParameter("name", str, "Name"),
            FunctionParameter("scale", float, "Scale", False),
        ],
        function=configured_tool,
    )
    ToolRegistry.tools["configured"] = tool

    function, bound_args = apply_bound_args(
        "configured",
        {
            "config": {"prefix": "p", "count": 2},
            "scale": "2.5",
        },
    )

    assert bound_args["config"] == BoundConfig(prefix="p", count=2)
    assert bound_args["scale"] == 2.5
    assert function(name="item") == "p:item:5.0"


def test_agent_info_resolves_tools_and_serializes_bound_args():
    ToolRegistry.tools.clear()

    def configured_tool(config: BoundConfig, name: str):
        return f"{config.prefix}:{name}"

    tool = ToolDescription(
        name="configured",
        description="Configured tool",
        parameters=[
            FunctionParameter("config", BoundConfig, "Config"),
            FunctionParameter("name", str, "Name"),
        ],
        function=configured_tool,
    )
    ToolRegistry.tools["configured"] = tool

    agent_info = AgentInfo(
        prompt_settings={
            "base_prompt": {"system": "sys", "novel_instruction": "ask"},
            "output_type": "SLDP",
        },
        tools=[
            {
                "name": "configured",
                "bound_args": {"config": {"prefix": "p", "count": 3}},
            }
        ],
        tool_interface="custom",
        max_iterations=2,
    )

    resolved_tool = agent_info.tools["configured"]
    assert resolved_tool is not tool
    assert resolved_tool.function(name="x") == "p:x"
    assert tool._bound_args is None
    assert agent_info.model_dump()["tools"] == [
        {"name": "configured", "bound_args": {"config": {"prefix": "p", "count": 3}}}
    ]


def test_agent_info_rejects_unknown_tools():
    ToolRegistry.tools.clear()

    with pytest.raises(ValueError, match="Unknown tool missing"):
        AgentInfo(
            prompt_settings={
                "base_prompt": {"system": "sys", "novel_instruction": "ask"}
            },
            tools=[{"name": "missing"}],
            tool_interface="custom",
            max_iterations=1,
        )


def test_in_context_example_provider_formats():
    example = InContextExample(user="u", assistant="a", system="s")

    assert render_openai_example(example) == [
        {"role": "developer", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert render_bedrock_example(example) == [
        {"role": "user", "content": [{"text": "u"}]},
        {"role": "assistant", "content": [{"text": "a"}]},
    ]


def test_prompt_loads_yaml_descriptions_and_renders_provider_payloads(tmp_path):
    description_file = tmp_path / "descriptions.yaml"
    description_file.write_text(
        yaml.safe_dump(
            {
                "scene_graph_description": "scene",
                "interface_description": "interface",
                "domain_description": "domain",
                "labelspace_description": "labels",
                "in_context_examples": [
                    {"user": "example user", "assistant": "example answer"}
                ],
            }
        ),
        encoding="utf-8",
    )

    prompt = Prompt(
        system="system",
        scene_graph_description=str(description_file),
        interface_description=str(description_file),
        domain_description=str(description_file),
        labelspace_description=str(description_file),
        tool_description="tools",
        in_context_examples_preamble="examples",
        in_context_examples=str(description_file),
        novel_instruction_preamble="now",
        novel_instruction="question",
        answer_semantic_guidance="semantic",
        answer_formatting_guidance="format",
    )
    prompt.set_api_prompt("api")

    openai = render_openai_prompt(prompt)
    assert openai[0] == {"role": "developer", "content": "system"}
    assert {"role": "developer", "content": "scene"} in openai
    assert {"role": "developer", "content": "api"} in openai
    assert {"role": "user", "content": "example user"} in openai
    assert {"role": "assistant", "content": "example answer"} in openai
    assert openai[-3:] == [
        {"role": "user", "content": "question"},
        {"role": "developer", "content": "semantic"},
        {"role": "developer", "content": "format"},
    ]

    anthropic = render_anthropic_prompt(prompt, novel_instruction="override")
    assert anthropic[0] == {"role": "user", "content": "system"}
    assert {"role": "user", "content": "override"} in anthropic

    bedrock = render_bedrock_prompt(prompt, novel_instruction="override")
    assert bedrock[0] == {"role": "user", "content": [{"text": "system"}]}
    assert {"role": "user", "content": [{"text": "override"}]} in bedrock


def test_pddl_examples_are_one_user_message_for_benchmark_providers(monkeypatch):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("HERACLES_AGENTS_PATH", str(project_root))
    settings = PromptSettings(
        base_prompt=str(
            project_root
            / "examples"
            / "prompts"
            / "cypher"
            / "pddl_agentic_cypher_prompt.yaml"
        )
    )
    examples = settings.base_prompt.in_context_examples

    assert isinstance(examples, str)
    assert examples.startswith("<PDDL Examples>")
    assert examples.endswith("</PDDL Examples>\n")

    for render in (render_ollama_prompt, render_openrouter_prompt):
        messages = render(settings.base_prompt, novel_instruction="Test instruction")
        example_messages = [message for message in messages if message["content"] == examples]
        assert example_messages == [{"role": "user", "content": examples}]
        assert all(message["role"] == "user" for message in messages)


def test_prompt_requires_novel_instruction_and_valid_yaml_keys(tmp_path):
    with pytest.raises(ValueError, match="novel_instruction must be set"):
        render_openai_prompt(Prompt(system="system"))

    missing_file = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match="Description YAML path does not exist"):
        Prompt(system="system", scene_graph_description=str(missing_file))

    wrong_keys_file = tmp_path / "wrong.yaml"
    wrong_keys_file.write_text(yaml.safe_dump({"other": "value"}), encoding="utf-8")
    prompt = Prompt(
        system="system",
        novel_instruction="question",
        scene_graph_description=str(wrong_keys_file),
    )
    assert prompt.scene_graph_description is None


def test_prompt_settings_loads_path_dict_and_prompt(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompt.yaml"
    prompt_file.write_text(
        yaml.safe_dump({"system": "sys", "novel_instruction": "ask"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPT_FILE", str(prompt_file))

    from_path = PromptSettings(base_prompt="$PROMPT_FILE", output_type="SLDP")
    assert from_path.base_prompt.system == "sys"
    assert from_path.output_type == "SLDP"
    assert not from_path.include_answer_type_hint

    with_answer_type_hint = PromptSettings(
        base_prompt="$PROMPT_FILE",
        output_type="SLDP",
        answer_type_hint=True,
    )
    assert with_answer_type_hint.include_answer_type_hint

    with pytest.raises(ValueError, match="sldp_answer_type_hint"):
        PromptSettings(
            base_prompt="$PROMPT_FILE",
            output_type="SLDP",
            sldp_answer_type_hint=True,
        )

    from_dict = PromptSettings(base_prompt={"system": "dict", "novel_instruction": "ask"})
    assert from_dict.base_prompt.system == "dict"

    existing_prompt = Prompt(system="existing", novel_instruction="ask")
    from_prompt = PromptSettings(base_prompt=existing_prompt)
    assert from_prompt.base_prompt is existing_prompt

    with pytest.raises(ValueError, match="Prompt path does not exist"):
        PromptSettings(base_prompt=str(tmp_path / "missing.yaml"))
    with pytest.raises(ValueError, match="cannot initialize base_prompt"):
        PromptSettings(base_prompt=42)


def test_answer_guidance_helpers_return_expected_sections():
    assert "SLDP Language" in get_sldp_format_description()
    assert "<answer>" in get_sldp_answer_tag_text()
    assert "PDDL Goal Language" in get_pddl_format_description()
    assert "<answer>" in get_pddl_answer_tag_text()
