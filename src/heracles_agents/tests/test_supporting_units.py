from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from heracles_agents.agent_functions import (
    build_custom_tool_prompt,
    call_custom_tool_from_string,
    extract_answer_tag,
    extract_tag,
    generate_update_for_history,
    get_tool_function,
    get_text_body,
    is_custom_tool_call,
)
from heracles_agents.llm_interface import (
    generate_tools_for_agent,
    get_summary_text,
)
from heracles_agents.provider_integrations.bedrock.bedrock_agent_integration import (
    BedrockMessage,
    get_bedrock_block_summary,
)
from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.cli.summarize_results import (
    colorize,
    flatten_analysis_dict,
    generate_table,
    summarize_results,
    to_string,
)
from heracles_agents.token_utils import count_text_tokens, estimate_text_tokens
from heracles_agents.provider_integrations.openai.token_counting import (
    count_openai_text_tokens,
)
from heracles_agents.provider_integrations.openrouter.token_counting import (
    count_openrouter_text_tokens,
)
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription
from heracles_agents.tool_calling.rendering import render_custom_tool
from heracles_agents.provider_integrations.anthropic.tool_rendering import render_anthropic_tool
from heracles_agents.provider_integrations.bedrock.tool_rendering import render_bedrock_tool
from heracles_agents.provider_integrations.ollama.tool_rendering import render_ollama_tool
from heracles_agents.provider_integrations.openai.tool_rendering import render_openai_tool
from heracles_agents.provider_integrations.openrouter.tool_rendering import render_openrouter_tool
from heracles_agents.tool_calling.registry import ToolRegistry, register_tool
from heracles_agents.tools.answer_tool import answer_tool
from heracles_agents.tools.calculator_tool import test_calculator as calculator_fn
from heracles_agents.tools.canary_favog_tool import the_mighty_favog
from heracles_agents.tools.codegen_tool import execute_generated_code
from heracles_agents.tools.cypher_query_tool import bind_query_db, query_db
from heracles_agents.tools.pddl_calling_tool import send_pddl
from heracles_agents.tools.pddl_answer_tool import pddl_tool


def test_structured_tool_openai_format_and_unsupported_providers():
    tool = StructuredToolDescription(
        name="structured_answer",
        description="Return a constrained answer",
        grammar="start: WORD",
    )

    assert render_openai_tool(tool) == {
        "type": "custom",
        "name": "structured_answer",
        "description": "Return a constrained answer",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: WORD",
        },
    }

    for renderer in [
        render_anthropic_tool,
        render_ollama_tool,
        render_bedrock_tool,
        render_openrouter_tool,
        render_custom_tool,
    ]:
        with pytest.raises(NotImplementedError):
            renderer(tool)

    rendered_pddl_tool = render_openai_tool(pddl_tool)
    assert rendered_pddl_tool["name"] == "pddl_answer_tool"
    assert rendered_pddl_tool["format"]["syntax"] == "lark"
    assert "fact:" in rendered_pddl_tool["format"]["definition"]


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


def test_provider_aware_token_counting_uses_openai_alias_and_estimate_fallback():
    encoder = Mock()
    encoder.encode.side_effect = lambda text: list(text)

    with patch(
        "heracles_agents.provider_integrations.openai.token_counting.tiktoken.encoding_for_model",
        return_value=encoder,
    ) as by_model:
        assert count_openai_text_tokens("gpt-5.4-mini", "abc") == 3
        by_model.assert_called_once_with("gpt-5-latest")

    with patch(
        "heracles_agents.provider_integrations.openai.token_counting.tiktoken.encoding_for_model",
        side_effect=KeyError("unknown"),
    ):
        assert count_openai_text_tokens("unknown-model", "hello world") == estimate_text_tokens(
            "hello world"
        )

    with patch(
        "heracles_agents.provider_integrations.openai.token_counting.tiktoken.encoding_for_model",
        return_value=encoder,
    ) as by_model:
        assert count_openrouter_text_tokens("openai/gpt-5.4-mini", "abcd") == 4
        by_model.assert_called_once_with("gpt-5-latest")

    agent = SimpleNamespace(
        client=SimpleNamespace(client_type="unknown"),
        model_info=SimpleNamespace(model="unknown-model"),
    )
    assert count_text_tokens(agent, "hello world") == estimate_text_tokens("hello world")


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
    assert to_string({"nested": "value"}) == "{'nested': 'value'}"


def test_flatten_analysis_dict_expands_latency_metrics():
    flattened = flatten_analysis_dict(
        {
            "correct": True,
            "latency": {
                "end_to_end_seconds": 1.2,
                "llm_call_seconds": 1.0,
            },
            "cost": {
                "total_cost_usd": 0.001,
            },
        }
    )

    assert flattened == {
        "correct": True,
        "latency_e2e_s": 1.2,
        "latency_llm_s": 1.0,
        "latency_tool_s": None,
        "latency_neo4j_s": None,
        "latency_validation_s": None,
        "cost_total_usd": 0.001,
    }


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
    prompt = build_custom_tool_prompt(
        [
            ToolDescription(
                name="echo",
                description="Echo",
                parameters=[FunctionParameter("value", str, "Value")],
                function=lambda value: value,
            )
        ]
    )
    assert "The following tools can be used" in prompt
    assert "Function name: echo" in prompt

    assert call_custom_tool_from_string(tools, "echo(value='a\\\"b')") == 'a"b'
    assert "Improperly formatted" in call_custom_tool_from_string(tools, "not a call")
    assert get_tool_function(tools, "echo")("x") == "x"
    with pytest.raises(ValueError, match="Unknown tool 'missing'.*echo"):
        get_tool_function(tools, "missing")
    with pytest.raises(ValueError, match="Unknown tool 'missing'.*echo"):
        call_custom_tool_from_string(tools, "missing(value='x')")

    assert extract_tag("answer", "x<answer>first</answer><answer>second</answer>") == "second"
    assert extract_tag("answer", "x<answer>first<answer>second</answer>") == "second"
    assert extract_answer_tag("<answer>done</answer>") == "done"
    assert extract_answer_tag("missing") is None

    assert get_text_body({"text": "hello"}) == "hello"
    assert get_text_body({"content": "body"}) == "body"
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
    assert get_summary_text(
        BedrockMessage({"toolUse": {"name": "lookup", "input": {"x": 1}}})
    ) == (
        "Function Call: lookup(x=1,)"
    )
    assert get_summary_text({"role": "assistant", "content": [{"text": "hi"}]}) == (
        "assistant:hi"
    )
    assert get_summary_text({"role": "assistant", "content": [{"content": "hi"}]}) == (
        "assistant:{'content': 'hi'}"
    )
    assert get_summary_text({"unexpected": "value"}) == "{'unexpected': 'value'}"

    assert get_bedrock_block_summary(BedrockMessage({"text": "hi"})) == "hi"
    assert get_bedrock_block_summary(BedrockMessage({"image": "unsupported"})) == (
        "{'image': 'unsupported'}"
    )


def test_generate_tools_for_agent_dispatches_interfaces():
    def echo(value: str):
        return value

    tool = ToolDescription(
        name="echo",
        description="Echo",
        parameters=[FunctionParameter("value", str, "Value")],
        function=echo,
    )

    for interface in ["openai", "anthropic", "ollama", "bedrock", "openrouter"]:
        agent_info = SimpleNamespace(tool_interface=interface, tools={"tool": tool})
        rendered = generate_tools_for_agent(agent_info)
        assert str(rendered[0]).count("echo") >= 1

    for interface in ["custom", "none"]:
        agent_info = SimpleNamespace(tool_interface=interface, tools={"tool": tool})
        assert generate_tools_for_agent(agent_info) == []

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

    calls = []

    def fake_query_db(cypher_string, dsgdb_conf=None):
        calls.append((cypher_string, dsgdb_conf))
        return "result"

    monkeypatch.setattr(
        "heracles_agents.tools.cypher_query_tool.query_db",
        fake_query_db,
    )
    dsgdb_conf = object()
    assert bind_query_db(dsgdb_conf)("MATCH (n) RETURN n") == "result"
    assert calls == [("MATCH (n) RETURN n", dsgdb_conf)]

    assert execute_generated_code("x = 1") == (
        "'solve_task' function not found in the generated code."
    )
    assert execute_generated_code("def solve_task(G):\n    return len(G)", SimpleNamespace(get_dsg=lambda: [1, 2])) == 2
    assert "boom" in execute_generated_code("raise RuntimeError('boom')")
