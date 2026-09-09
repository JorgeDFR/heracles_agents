from types import SimpleNamespace

from heracles_agents.llm_interface import EvalQuestion
from heracles_agents.pipelines.cypher_validation import validate_last_cypher_tool_call


def question(solution, comparison_type="SLDP", text="Question"):
    return EvalQuestion(
        uid="q1",
        name="Validation",
        question=text,
        solution=solution,
        correctness_comparator={
            "comparison_type": comparison_type,
            "relation": "equal",
        },
    )


def context(*calls):
    return SimpleNamespace(tool_executions=list(calls))


def test_last_cypher_call_drives_qa_solution_and_execution_validation():
    result = validate_last_cypher_tool_call(
        question("65"),
        [
            context(
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "bad"},
                    "output": "syntax error",
                    "rows": None,
                    "executable": False,
                    "error": "syntax error",
                },
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "MATCH (o:Object) RETURN count(o) AS count"},
                    "output": "[{'count': 65}]",
                    "rows": [{"count": 65}],
                    "executable": True,
                    "error": None,
                },
            )
        ],
    )

    assert result["tool_executable"] is True
    assert result["cypher_solution_match"] is True
    assert result["generated_cypher"].endswith("AS count")
    assert result["cypher_validation_issues"] == []


def test_pddl_validation_compares_cypher_derived_grounding_symbols():
    result = validate_last_cypher_tool_call(
        question(
            "(or (holding O237) (holding O300))",
            "PDDL",
            "Bring along any light you can choose.",
        ),
        [
            context(
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "MATCH ..."},
                    "output": "[{'nodeSymbol': 'O237'}, {'nodeSymbol': 'O300'}]",
                    "rows": [{"nodeSymbol": "O237"}, {"nodeSymbol": "O300"}],
                    "executable": True,
                    "error": None,
                }
            )
        ],
    )

    assert result["cypher_solution_match"] is True


def test_missing_cypher_call_is_not_counted_as_execution_failure():
    result = validate_last_cypher_tool_call(question("1"), [context()])

    assert result["tool_executable"] is None
    assert result["cypher_solution_match"] is None


def test_qa_validation_preserves_collected_values_for_nested_sldp():
    call = {
        "tool_name": "run_cypher_query",
        "tool_args": {"cypher_string": "MATCH ... RETURN collect(o.nodeSymbol)"},
        "output": "[{'objects': ['O59', 'O63']}]",
        "rows": [{"objects": ["O59", "O63"]}],
        "executable": True,
        "error": None,
    }

    result = validate_last_cypher_tool_call(
        question("<[O59, O63]>", text="Return grouped objects."), [context(call)]
    )

    assert result["cypher_solution_match"] is True
