"""Execution details recorded for Cypher tool calls."""

from typing import Any


def summarize_last_cypher_tool_call(contexts) -> dict[str, Any]:
    """Return the last Cypher call's status, query, and output."""
    calls = [
        call
        for context in contexts
        for call in getattr(context, "tool_executions", [])
        if call.get("tool_name") == "run_cypher_query"
    ]
    if not calls:
        return {
            "tool_executable": None,
            "generated_cypher": None,
            "cypher_tool_output": None,
        }

    call = calls[-1]
    return {
        "tool_executable": bool(call.get("executable")),
        "generated_cypher": (call.get("tool_args") or {}).get("cypher_string"),
        "cypher_tool_output": call.get("output"),
    }
