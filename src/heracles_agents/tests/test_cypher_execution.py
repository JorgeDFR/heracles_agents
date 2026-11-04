from types import SimpleNamespace

from heracles_agents.pipelines.cypher_execution import summarize_last_cypher_tool_call


def test_execution_summary_uses_last_cypher_call_across_contexts():
    result = summarize_last_cypher_tool_call([
        SimpleNamespace(tool_executions=[{
            "tool_name": "run_cypher_query",
            "tool_args": {"cypher_string": "bad"},
            "output": "syntax error",
            "executable": False,
        }]),
        SimpleNamespace(tool_executions=[
            {
                "tool_name": "run_cypher_query",
                "tool_args": {"cypher_string": "MATCH (n) RETURN n"},
                "output": "[{'n': 'O1'}]",
                "executable": True,
            },
            {"tool_name": "submit_answer", "output": "O2"},
        ]),
    ])

    assert result == {
        "tool_executable": True,
        "generated_cypher": "MATCH (n) RETURN n",
        "cypher_tool_output": "[{'n': 'O1'}]",
    }


def test_execution_summary_preserves_failed_call_output():
    result = summarize_last_cypher_tool_call([
        SimpleNamespace(tool_executions=[{
            "tool_name": "run_cypher_query",
            "tool_args": {"cypher_string": "bad"},
            "output": "syntax error",
            "executable": False,
        }]),
    ])

    assert result == {
        "tool_executable": False,
        "generated_cypher": "bad",
        "cypher_tool_output": "syntax error",
    }


def test_execution_summary_without_cypher_calls_is_unknown():
    assert summarize_last_cypher_tool_call([
        SimpleNamespace(),
        SimpleNamespace(tool_executions=[{"tool_name": "submit_answer"}]),
    ]) == {
        "tool_executable": None,
        "generated_cypher": None,
        "cypher_tool_output": None,
    }
