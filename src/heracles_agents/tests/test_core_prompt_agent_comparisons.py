from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from heracles_agents.llm_agent import AgentInfo, apply_bound_args
from heracles_agents.llm_interface import PddlComparison, SldpComparison
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.prompt import (
    InContextExample,
    Prompt,
    PromptSettings,
    get_sldp_answer_tag_text,
    get_sldp_format_description,
)
from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry


class BoundConfig(BaseModel):
    prefix: str
    count: int


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

    assert example.to_openai_json() == [
        {"role": "developer", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert example.to_bedrock_json() == [
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

    openai = prompt.to_openai_json()
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

    anthropic = prompt.to_anthropic_json(novel_instruction="override")
    assert anthropic[0] == {"role": "user", "content": "system"}
    assert {"role": "user", "content": "override"} in anthropic

    bedrock = prompt.to_bedrock_json(novel_instruction="override")
    assert bedrock[0] == {"role": "user", "content": [{"text": "system"}]}
    assert {"role": "user", "content": [{"text": "override"}]} in bedrock


def test_prompt_requires_novel_instruction_and_valid_yaml_keys(tmp_path):
    with pytest.raises(ValueError, match="novel_instruction must be set"):
        Prompt(system="system").to_openai_json()

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

    from_dict = PromptSettings(base_prompt={"system": "dict", "novel_instruction": "ask"})
    assert from_dict.base_prompt.system == "dict"

    existing_prompt = Prompt(system="existing", novel_instruction="ask")
    from_prompt = PromptSettings(base_prompt=existing_prompt)
    assert from_prompt.base_prompt is existing_prompt

    with pytest.raises(ValueError, match="Prompt path does not exist"):
        PromptSettings(base_prompt=str(tmp_path / "missing.yaml"))
    with pytest.raises(ValueError, match="cannot initialize base_prompt"):
        PromptSettings(base_prompt=42)


def test_sldp_guidance_helpers_return_expected_sections():
    assert "SLDP Equality Language" in get_sldp_format_description()
    assert "<answer>" in get_sldp_answer_tag_text()
