# ruff: noqa: F811
import logging
import re

from plum import dispatch

from heracles_agents.llm_interface import PddlComparison, SldpComparison
from pypddl.pddl_goal_manipulations import pddl_goal_equals
from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from pypddl.pddl_goal_types import (
    Bool,
    Conjunction,
    Disjunction,
    Fact,
    NegatedAtomic,
    NegatedClause,
)
from sldp.sldp_lang import lark_parse_sldp, sldp_equals

logger = logging.getLogger(__name__)


_PDDL_PREDICATES = {
    "visited-object": (r"O\d+",),
    "at-object": (r"O\d+",),
    "visited-region": (r"R\d+",),
    "in-region": (r"R\d+",),
    "visited-place": (r"[Pp]\d+",),
    "at-place": (r"[Pp]\d+",),
    "safe": (r"O\d+",),
    "holding": (r"O\d+",),
    "object-in-place": (r"O\d+", r"[Pp]\d+"),
}


def _valid_grounded_pddl_goal(goal) -> bool:
    """Validate facts against the predicates exposed by the benchmark domain."""
    if isinstance(goal, Fact):
        parameter_patterns = _PDDL_PREDICATES.get(goal.head)
        return (
            parameter_patterns is not None
            and len(goal.params) == len(parameter_patterns)
            and all(
                re.fullmatch(pattern, parameter) is not None
                for pattern, parameter in zip(parameter_patterns, goal.params)
            )
        )
    if isinstance(goal, (Conjunction, Disjunction)):
        return all(_valid_grounded_pddl_goal(clause) for clause in goal.clauses)
    if isinstance(goal, NegatedAtomic):
        return isinstance(goal.atomic, Fact) and _valid_grounded_pddl_goal(goal.atomic)
    if isinstance(goal, NegatedClause):
        return _valid_grounded_pddl_goal(goal.clause)
    return isinstance(goal, Bool)


def _normalized_singleton_set_answer(answer, solution):
    """Return a narrowly repaired SLDP singleton-set answer, if applicable."""
    try:
        solution_ast = lark_parse_sldp(solution)
    except Exception:
        return None
    if not (
        isinstance(solution_ast, tuple)
        and solution_ast[:1] == ("set",)
        and len(solution_ast) == 2
    ):
        return None

    text = str(answer).strip()
    candidates = []
    if text.startswith("<") and not text.endswith(">"):
        candidates.append(text + ">")
    elif text and not text.startswith("<"):
        candidates.append(f"<{text}>")

    for candidate in candidates:
        try:
            lark_parse_sldp(candidate)
        except Exception:
            continue
        if sldp_equals(solution, candidate):
            return candidate
    return None


@dispatch
def evaluate_answer(comparator: PddlComparison, answer, solution):
    try:
        parsed_goal = lark_parse_pddl_goal(answer)
        valid_pddl = _valid_grounded_pddl_goal(parsed_goal)
        if not valid_pddl:
            logger.warning(
                "PDDL goal uses an unknown predicate, invalid arity, "
                "or invalid symbol type"
            )
    except Exception as ex:
        logger.warning("Invalid PDDL goal")
        logger.warning(str(ex))
        valid_pddl = False

    if valid_pddl:
        correct = pddl_goal_equals(parsed_goal, lark_parse_pddl_goal(solution))
    else:
        correct = False

    return valid_pddl, correct


@dispatch
def evaluate_answer(comparator: SldpComparison, answer, solution):
    try:
        lark_parse_sldp(answer)
        valid_sldp = True
    except Exception as ex:
        logger.warning("Invalid SLDP")
        logger.warning(str(ex))
        valid_sldp = False

    correct = valid_sldp and sldp_equals(solution, answer)
    if not correct:
        normalized = _normalized_singleton_set_answer(answer, solution)
        correct = normalized is not None
    return valid_sldp, correct
