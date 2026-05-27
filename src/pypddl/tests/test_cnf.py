from pypddl.pddl_goal_manipulations import convert_to_cnf
from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from pypddl.pddl_goal_types import clause_equals


def assert_cnf_equal(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    cnf = convert_to_cnf(parsed_formula)
    assert clause_equals(cnf, parsed_expected), (
        f"Got {str(cnf)}, Expected {str(parsed_expected)}"
    )


def test_single_variable():
    assert_cnf_equal("?a", "?a")


def test_simple_or():
    assert_cnf_equal("(or ?a ?b)", "(or ?a ?b)")


def test_negation_pushdown():
    assert_cnf_equal("(not (or ?a ?b))", "(and (not ?a) (not ?b))")


def test_distribution():
    assert_cnf_equal("(or ?a (and ?b ?c))", "(and (or ?a ?b) (or ?a ?c))")


def test_deeply_nested_expression():
    formula = "(or (and ?a ?b) (and ?c ?d))"
    expected = "(and (or ?a ?c) (or ?a ?d) (or ?b ?c) (or ?b ?d))"
    assert_cnf_equal(formula, expected)
