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
                    "tool_args": {
                        "cypher_string": "MATCH (o:Object) RETURN count(o) AS count"
                    },
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


def test_pddl_validation_aggregates_grounding_across_calls():
    result = validate_last_cypher_tool_call(
        question(
            "(and (or (visited-object O59) (visited-object O291)) (in-region R5))",
            "PDDL",
            "Visit a carton, then finish in R5.",
        ),
        [
            context(
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "objects"},
                    "output": "[{'id': 'O59'}, {'id': 'O291'}]",
                    "rows": [{"id": "O59"}, {"id": "O291"}],
                    "executable": True,
                    "error": None,
                },
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "room"},
                    "output": "[{'id': 'R5'}]",
                    "rows": [{"id": "R5"}],
                    "executable": True,
                    "error": None,
                },
            )
        ],
    )

    assert result["cypher_solution_match"] is True
    assert result["generated_cypher"] == "room"


def test_pddl_validation_ignores_superseded_executable_query():
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
                    "tool_args": {"cypher_string": "too broad"},
                    "output": "[{'id': 'O1'}, {'id': 'O237'}, {'id': 'O300'}]",
                    "rows": [{"id": "O1"}, {"id": "O237"}, {"id": "O300"}],
                    "executable": True,
                    "error": None,
                },
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "corrected"},
                    "output": "[{'id': 'O237'}, {'id': 'O300'}]",
                    "rows": [{"id": "O237"}, {"id": "O300"}],
                    "executable": True,
                    "error": None,
                },
            )
        ],
    )

    assert result["cypher_solution_match"] is True


def test_pddl_validation_does_not_require_grounding_for_explicit_symbols():
    result = validate_last_cypher_tool_call(
        question("(in-region R1)", "PDDL", "Finish in R1."),
        [
            context(
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "broad"},
                    "output": "[{'id': 'R1'}, {'id': 'R2'}]",
                    "rows": [{"id": "R1"}, {"id": "R2"}],
                    "executable": True,
                    "error": None,
                }
            )
        ],
    )

    assert result["cypher_solution_match"] is True


def test_pddl_validation_accepts_first_row_of_ranked_query():
    result = validate_last_cypher_tool_call(
        question(
            "(visited-region R2)",
            "PDDL",
            "Visit a room with the most distinct neighbors.",
        ),
        [
            context(
                {
                    "tool_name": "run_cypher_query",
                    "tool_args": {"cypher_string": "ranked"},
                    "output": "[{'room': 'R2'}, {'room': 'R3'}]",
                    "rows": [{"room": "R2", "degree": 3}, {"room": "R3", "degree": 2}],
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


def test_qa_validation_reconstructs_point_from_coordinate_columns():
    call = {
        "tool_name": "run_cypher_query",
        "tool_args": {"cypher_string": "MATCH ... RETURN p.x, p.y, p.z"},
        "output": "[{'x': 1.001, 'y': 2.002, 'z': 3.003}]",
        "rows": [{"x": 1.001, "y": 2.002, "z": 3.003}],
        "executable": True,
        "error": None,
    }

    result = validate_last_cypher_tool_call(
        question("POINT(1.00 2.00 3.00)"), [context(call)]
    )

    assert result["cypher_solution_match"] is True


def test_qa_validation_accepts_symbol_pair_in_first_ranked_row():
    call = {
        "tool_name": "run_cypher_query",
        "tool_args": {"cypher_string": "MATCH ... ORDER BY distance"},
        "output": "ranked pairs",
        "rows": [
            {"left": "R4", "right": "R5", "distance": 1.0},
            {"left": "R2", "right": "R3", "distance": 2.0},
        ],
        "executable": True,
        "error": None,
    }

    result = validate_last_cypher_tool_call(
        question("<[R4, R5]>", text="Return the nearest room pair."), [context(call)]
    )

    assert result["cypher_solution_match"] is True
