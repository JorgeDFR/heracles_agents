from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from heracles_agents.agent_functions import (
    call_custom_tool_from_string,
    extract_answer_tag,
    extract_tag,
    generate_update_for_history,
    get_text_body,
    is_custom_tool_call,
)
from heracles_agents.llm_interface import (
    generate_tools_for_agent,
    get_bedrock_block_summary,
    get_summary_text,
)
from heracles_agents.structured_tool_interface import StructuredToolDescription
from heracles_agents.summarize_results import (
    colorize,
    generate_table,
    summarize_results,
    to_string,
)
from heracles_agents.token_utils import get_token_encoder
from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool
from heracles_agents.tools.answer_tool import answer_tool
from heracles_agents.tools.calculator_tool import test_calculator as calculator_fn
from heracles_agents.tools.canary_favog_tool import the_mighty_favog
from heracles_agents.tools.codegen_tool import execute_generated_code
from heracles_agents.tools.cypher_query_tool import query_db
from heracles_agents.tools.pddl_calling_tool import send_pddl


def test_structured_tool_openai_format_and_unsupported_providers():
    tool = StructuredToolDescription(
        name="structured_answer",
        description="Return a constrained answer",
        grammar="start: WORD",
    )

    assert tool.to_openai_responses() == {
        "type": "custom",
        "name": "structured_answer",
        "description": "Return a constrained answer",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: WORD",
        },
    }

    for method in [
        tool.to_anthropic,
        tool.to_ollama,
        tool.to_bedrock,
        tool.to_openrouter,
        tool.to_custom,
    ]:
        with pytest.raises(NotImplementedError):
            method()


def test_tool_registry_registers_duplicates_and_validates_arg_types(capsys):
    ToolRegistry.tools.clear()

    def sample_tool(name: str, count=None):
        return name, count

    tool = ToolDescription(
        name="sample",
        description="Sample tool",
        parameters=[
            FunctionParameter("name", str, "Name"),
        ],
        function=sample_tool,
    )

    register_tool(tool)
    register_tool(tool)

    assert ToolRegistry.registered_tool_summary() == ["sample"]
    assert "sample already has registered function!" in capsys.readouterr().out
    assert ToolRegistry.get_arg_type("sample", "name") is str

    with pytest.raises(ValueError, match="not present"):
        ToolRegistry.get_arg_type("missing", "name")
    with pytest.raises(ValueError, match="not a valid argument"):
        ToolRegistry.get_arg_type("sample", "missing")
    with pytest.raises(ValueError, match="does not have a type"):
        ToolRegistry.get_arg_type("sample", "count")


def test_token_encoder_uses_gpt5_alias_and_fallback():
    alias_encoder = object()
    fallback_encoder = object()

    with (
        patch("heracles_agents.token_utils.tiktoken.encoding_for_model") as by_model,
        patch("heracles_agents.token_utils.tiktoken.get_encoding") as get_encoding,
    ):
        by_model.side_effect = [alias_encoder, KeyError("unknown")]
        get_encoding.return_value = fallback_encoder

        assert get_token_encoder("gpt-5.4-mini") is alias_encoder
        assert by_model.call_args_list[0].args == ("gpt-5-latest",)
        assert get_token_encoder("unknown-model") is fallback_encoder
        get_encoding.assert_called_once_with("cl100k_base")


def test_summarize_results_counts_numeric_fields_and_formats_values():
    questions = [
        {"correct": True, "score": 2, "ratio": 0.5, "name": "q1"},
        {"correct": False, "score": 3, "ratio": 0.25, "name": "q2"},
    ]

    ratios, strings = summarize_results(questions)

    assert ratios == {"correct": 0.5, "score": 2.5, "ratio": 0.375, "questions": 2}
    assert strings == {
        "correct": "1/2",
        "score": "5/2",
        "ratio": "0.75/2",
        "questions": "2",
    }
    assert colorize("green", "ok") == "[green]ok[/green]"
    assert to_string(True) == "[green]True[/green]"
    assert to_string(False) == "[red]False[/red]"
    assert to_string(3) == "3"
    assert to_string(1.25) == "1.25"
    assert to_string("raw") == "raw"


def test_generate_table_uses_remapped_columns_first():
    table = generate_table(
        "Results",
        [{"name": "q1", "question": "How?", "correct": True}],
        {"Name": "name"},
    )

    assert table.title == "Results"
    assert [column.header for column in table.columns] == [
        "Name",
        "question",
        "correct",
    ]


def test_agent_function_helpers_and_custom_tool_calls():
    tools = {
        "echo": SimpleNamespace(function=lambda value: value),
    }
    assert call_custom_tool_from_string(tools, "echo(value='a\\\"b')") == 'a"b'
    assert "Improperly formatted" in call_custom_tool_from_string(tools, "not a call")

    assert extract_tag("answer", "x<answer>first</answer><answer>second</answer>") == "second"
    assert extract_answer_tag("<answer>done</answer>") == "done"
    assert extract_answer_tag("missing") is None

    assert get_text_body({"text": "hello"}) == "hello"
    assert get_text_body({"content": "body"}) == "body"
    assert get_text_body({"toolUse": {"name": "fn", "input": {"a": 1}}}) == "fn(a=1,)"
    with pytest.raises(NotImplementedError):
        get_text_body({"unknown": "shape"})

    agent = SimpleNamespace(agent_info=SimpleNamespace(tool_interface="custom"))
    assert is_custom_tool_call(agent, {"content": "<tool> echo(value='x') </tool>"})
    assert not is_custom_tool_call(agent, {"content": "plain text"})
    assert not is_custom_tool_call(
        SimpleNamespace(agent_info=SimpleNamespace(tool_interface="openai")),
        {"content": "<tool> echo(value='x') </tool>"},
    )

    assert generate_update_for_history(object(), ["message"]) == ["message"]


def test_llm_summary_helpers_cover_provider_shapes():
    assert get_summary_text("plain") == "plain"
    assert get_summary_text(None) == ""
    assert get_summary_text(["a", None, {"role": "assistant", "content": "b"}]) == (
        "a\n\nassistant: b"
    )
    assert get_summary_text({"type": "function_call_output", "output": "ok"}) == (
        "Function result: ok"
    )
    assert get_summary_text({"toolResult": {"content": [{"text": "ok"}]}}) == (
        "Tool result: ok"
    )
    assert get_summary_text({"toolUse": {"name": "lookup", "input": {"x": 1}}}) == (
        "Function Call: lookup(x=1,)"
    )
    assert get_summary_text({"role": "assistant", "content": [{"text": "hi"}]}) == (
        "assistant:hi"
    )
    assert get_summary_text({"role": "assistant", "content": [{"content": "hi"}]}) == (
        "assistant:{'content': 'hi'}"
    )
    assert get_summary_text({"unexpected": "value"}) == "{'unexpected': 'value'}"

    assert get_bedrock_block_summary({"text": "hi"}) == "hi"
    with pytest.raises(NotImplementedError):
        get_bedrock_block_summary({"image": "unsupported"})


def test_generate_tools_for_agent_dispatches_interfaces():
    tool = Mock()
    tool.to_openai_responses.return_value = "openai"
    tool.to_anthropic.return_value = "anthropic"
    tool.to_ollama.return_value = "ollama"
    tool.to_bedrock.return_value = "bedrock"
    tool.to_openrouter.return_value = "openrouter"

    for interface, expected in [
        ("openai", ["openai"]),
        ("anthropic", ["anthropic"]),
        ("ollama", ["ollama"]),
        ("bedrock", ["bedrock"]),
        ("openrouter", ["openrouter"]),
        ("custom", []),
        ("none", []),
    ]:
        agent_info = SimpleNamespace(tool_interface=interface, tools={"tool": tool})
        assert generate_tools_for_agent(agent_info) == expected

    with pytest.raises(NotImplementedError, match="Unknown tool interface"):
        generate_tools_for_agent(SimpleNamespace(tool_interface="bad", tools={}))


def test_simple_tool_functions_and_error_paths(monkeypatch):
    assert answer_tool("42") == "Final answer: 42"
    assert calculator_fn(4, 2, "add") == 6
    assert calculator_fn(4, 2, "subtract") == 2
    assert calculator_fn(4, 2, "multiply") == 8
    assert calculator_fn(4, 2, "divide") == 2
    assert calculator_fn(4, 0, "divide") == float("inf")
    with pytest.raises(ValueError, match="Unknown operation"):
        calculator_fn(4, 2, "power")

    assert the_mighty_favog("anything", "business") == 6
    assert the_mighty_favog("anything", "sports") == 4
    assert the_mighty_favog("anything", "personal") == 7
    assert the_mighty_favog("anything", "unknown") is None

    with pytest.raises(ValueError, match="robot_name or planner_topic missing"):
        send_pddl("(and)", robot_name=None, planner_topic="/planner")

    monkeypatch.setattr("heracles_agents.tools.pddl_calling_tool.os.system", lambda cmd: 0)
    assert send_pddl("(and)", robot_name="robot", planner_topic="/planner") == (
        "Sent goal (and) to robot robot on /planner"
    )

    with pytest.raises(ValueError, match="dsgdb_conf=None"):
        query_db("MATCH (n) RETURN n")

    assert execute_generated_code("x = 1") == (
        "'solve_task' function not found in the generated code."
    )
    assert execute_generated_code("def solve_task(G):\n    return len(G)", SimpleNamespace(get_dsg=lambda: [1, 2])) == 2
    assert "boom" in execute_generated_code("raise RuntimeError('boom')")
