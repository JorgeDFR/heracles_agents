from itertools import permutations

import pytest
from pypddl.pddl_goal_manipulations import (
    pddl_goal_equals,
)
from pypddl.pddl_goal_parser import lark_parse_pddl_goal


def assert_equal(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    assert pddl_goal_equals(parsed_formula, parsed_expected), (
        f"Got {str(parsed_formula)}, Expected {str(parsed_expected)}"
    )


def assert_unequal(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    assert not pddl_goal_equals(parsed_formula, parsed_expected), (
        f"Got {str(parsed_formula)}, expected to not equal {str(parsed_expected)}"
    )


def test_simple():
    assert_equal(
        "(and (visited-place p1) (visited-place p2))",
        "(and (visited-place p2) (visited-place p1))",
    )

    assert_equal(
        "(or (visited-place p1) (visited-place p2))",
        "(or (visited-place p2) (visited-place p1))",
    )

    assert_unequal(
        "(visited-place p1)",
        "(visited-place p2)",
    )

    assert_unequal(
        "(visited-place p1)",
        "(not (visited-place p1))",
    )


def test_simplification():
    assert_equal(
        "(not (not (visited-place p1)))",
        "(visited-place p1)",
    )


def test_demorgan():
    assert_equal("(not (and (vp p1) (vp p2)))", "(or (not (vp p1)) (not (vp p2)))")


PDDL039_SOLUTION = (
    "(and (or (safe O29) (safe O30)) "
    "(or (safe O87) (safe O92) (safe O97) (safe O102)))"
)


@pytest.mark.parametrize("order", list(permutations(["O87", "O92", "O97", "O102"])))
def test_pddl039_accepts_reordered_alternatives(order):
    alternatives = " ".join(f"(safe {object_id})" for object_id in order)
    answer = f"(and (or (safe O29) (safe O30)) (or {alternatives}))"

    assert_equal(answer, PDDL039_SOLUTION)
    assert_equal(PDDL039_SOLUTION, answer)


@pytest.mark.parametrize(
    "alternatives",
    [
        "(or (safe O87) (safe O92) (safe O102))",
        "(or (safe O87) (safe O92) (safe O97) (safe O101))",
        "(or (safe O87) (and (safe O92) (safe O97) (safe O102)))",
    ],
)
def test_pddl039_rejects_changed_alternatives(alternatives):
    answer = f"(and (or (safe O29) (safe O30)) {alternatives})"

    assert_unequal(answer, PDDL039_SOLUTION)
