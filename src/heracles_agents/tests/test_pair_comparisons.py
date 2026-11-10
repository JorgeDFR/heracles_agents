"""Pair ordering is a per-question validation policy, not a prompt requirement."""

import pytest
from heracles_agents.llm_interface import EvalQuestion, SldpComparison
from heracles_agents.pipelines.comparisons import evaluate_answer


@pytest.mark.parametrize(
    "solution,answer,expected",
    [
        ("[O1, O2]", "[O2, O1]", True),
        ("<[O1, O2], [O3, O4]>", "<[O4, O3], [O2, O1]>", True),
        ("{R1: <[O1, O2]>, R2: <[O3, O4]>}", "{R2: <[O4, O3]>, R1: <[O2, O1]>}", True),
        ("<[O1, O2]>", "<[O2, O3]>", False),
        ("<[O1, O2]>", "<[O1, O1]>", False),
        ("<[O1, O2]>", "<[O2, O1, O3]>", False),
        ("<[O1, O2], [O3, O4]>", "<[O2, O1]>", False),
        ("{R1: <[O1, O2]>}", "{R2: <[O2, O1]>}", False),
        ("<[O1, O2]>", "<<O2, O1>>", False),
        ("[1, 2]", "[2, 1]", False),
        ("[R1, R2, R3]", "[R3, R2, R1]", False),
    ],
)
def test_pair_aware_validation(solution, answer, expected):
    comparator = SldpComparison(
        comparison_type="SLDP", relation="equal", unordered_pairs=True
    )
    assert evaluate_answer(comparator, answer, solution) == (True, expected)


def test_route_order_remains_strict_without_pair_policy():
    comparator = SldpComparison(comparison_type="SLDP", relation="equal")
    assert not comparator.unordered_pairs
    assert evaluate_answer(comparator, "[R2, R1]", "[R1, R2]") == (True, False)


def test_pair_flag_survives_question_deserialization_and_singleton_repair():
    question = EvalQuestion.model_validate(
        {
            "uid": "pair",
            "name": "Pair",
            "question": "Which pair is closest?",
            "solution": "<[O1, O2]>",
            "correctness_comparator": {
                "comparison_type": "SLDP",
                "relation": "equal",
                "unordered_pairs": True,
            },
        }
    )
    assert question.correctness_comparator.unordered_pairs
    assert question.model_dump()["correctness_comparator"]["unordered_pairs"]
    assert evaluate_answer(
        question.correctness_comparator, "[O2, O1]", question.solution
    ) == (True, True)
