"""Validation of Cypher tool calls against a benchmark question."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from sldp.sldp_lang import lark_parse_sldp, sldp_equals

_SCENE_SYMBOL = re.compile(r"\b(?:O|R|P|M)\d+\b", re.IGNORECASE)


def validate_last_cypher_tool_call(question, contexts) -> dict[str, Any]:
    """Return independent execution and solution-grounding evidence.

    The last Cypher call remains authoritative for QA retries and for reported
    query/execution fields. PDDL grounding is accumulated across executable calls
    because separate calls may ground separate clauses of a compound goal. A
    missing call is represented as ``None`` rather than a failure.
    """

    calls = [
        call
        for context in contexts
        for call in getattr(context, "tool_executions", [])
        if call.get("tool_name") == "run_cypher_query"
    ]
    if not calls:
        return {
            "cypher_solution_match": None,
            "tool_executable": None,
            "generated_cypher": None,
            "cypher_tool_output": None,
            "cypher_validation_issues": [],
        }

    call = calls[-1]
    executable = bool(call.get("executable"))
    output = call.get("output")
    issues: list[str] = []
    match = False
    if not executable:
        issues.append(
            call.get("error") or "The final Cypher tool call was not executable."
        )
    else:
        rows = call.get("rows")
        comparison_type = getattr(
            question.correctness_comparator, "comparison_type", ""
        )
        if comparison_type == "PDDL":
            match = _pddl_grounding_matches_calls(
                calls, question.solution, question.question
            )
        else:
            match = _qa_rows_match(question, rows)
        if not match:
            issues.append(
                "The final Cypher tool output does not match the expected "
                "solution/grounding."
            )

    return {
        "cypher_solution_match": match if executable else False,
        "tool_executable": executable,
        "generated_cypher": (call.get("tool_args") or {}).get("cypher_string"),
        "cypher_tool_output": output,
        "cypher_validation_issues": issues,
    }


def _qa_rows_match(question, rows: Any) -> bool:
    for candidate in _solution_candidates(rows):
        try:
            lark_parse_sldp(candidate)
            matches = sldp_equals(question.solution, candidate)
        except Exception:
            continue
        if matches:
            return True
    expected_atoms = _flat_expected_atoms(question.solution)
    actual_atoms = {str(value).strip().casefold() for value in _atomic_values(rows)}
    # A query may return complete nodes alongside the requested property. Extra
    # properties do not invalidate grounding when every expected value is present.
    return bool(expected_atoms) and expected_atoms <= actual_atoms


def _flat_expected_atoms(solution: str) -> set[str]:
    text = str(solution).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
        return {item.strip().casefold() for item in text.split(",") if item.strip()}
    return {text.casefold()} if text else set()


def _pddl_grounding_matches_calls(
    calls: Sequence[Mapping[str, Any]],
    expected_solution: str,
    natural_language_question: str,
) -> bool:
    expected = {symbol.upper() for symbol in _SCENE_SYMBOL.findall(expected_solution)}
    explicit = {
        symbol.upper() for symbol in _SCENE_SYMBOL.findall(natural_language_question)
    }
    # Symbols stated directly by the user are not Cypher-derived grounding.
    # A direct-symbol-only goal therefore needs no database evidence.
    expected -= explicit
    if not expected:
        return True

    executable_rows = [
        call.get("rows") for call in calls if bool(call.get("executable"))
    ]
    call_symbols = [_symbols_in_rows(rows) - explicit for rows in executable_rows]
    # A sequence can contain both independent grounding calls and superseded
    # retries. Accept a subset of calls whose combined symbols exactly ground
    # the goal, without letting an unrelated broad query create a match.
    reachable = {frozenset()}
    for symbols in call_symbols:
        reachable |= {known.union(symbols) for known in tuple(reachable)}
    if frozenset(expected) in reachable:
        return True

    # Ranked queries commonly return the desired entity in their first row plus
    # lower-ranked alternatives. Preserve row order so that such evidence can be
    # recognized without accepting an unordered superset.
    return any(
        (_symbols_in_rows([rows[0]]) - explicit) == expected
        for rows in executable_rows
        if isinstance(rows, Sequence)
        and not isinstance(rows, (str, bytes, bytearray))
        and rows
    )


def _symbols_in_rows(rows: Any) -> set[str]:
    return {
        symbol.upper()
        for value in _atomic_values(rows)
        for symbol in _SCENE_SYMBOL.findall(str(value))
    }


def _solution_candidates(rows: Any) -> list[str]:
    atomic = _atomic_values(rows)
    candidates = [str(value) for value in atomic]
    if len(atomic) != 1:
        candidates.append("<" + ", ".join(str(value) for value in atomic) + ">")
        candidates.append("[" + ", ".join(str(value) for value in atomic) + "]")
    direct = _direct_row_values(rows)
    rendered_direct = [_sldp_render(value) for value in direct]
    candidates.extend(rendered_direct)
    if rendered_direct:
        candidates.append("<" + ", ".join(rendered_direct) + ">")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if all(axis in row for axis in ("x", "y", "z")):
                candidates.append(f"POINT({row['x']} {row['y']} {row['z']})")
            symbols = list(
                dict.fromkeys(
                    symbol.upper()
                    for value in _atomic_values(row)
                    for symbol in _SCENE_SYMBOL.findall(str(value))
                )
            )
            if symbols:
                symbol_list = "[" + ", ".join(symbols) + "]"
                candidates.extend(
                    [
                        "<" + ", ".join(symbols) + ">",
                        symbol_list,
                        "<" + symbol_list + ">",
                    ]
                )
    candidates.append(str(rows))
    return list(dict.fromkeys(candidates))


def _direct_row_values(rows: Any) -> list[Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return [rows]
    values = []
    for row in rows:
        if isinstance(row, Mapping):
            values.extend(row.values())
        else:
            values.append(row)
    return values


def _sldp_render(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_sldp_render(item) for item in value) + "]"
    if all(hasattr(value, coordinate) for coordinate in ("x", "y", "z")):
        return f"POINT({value.x} {value.y} {value.z})"
    return str(value)


def _atomic_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        preferred = [
            key
            for key in (
                "nodeSymbol",
                "symbol",
                "class",
                "semantic_label",
                "center",
                "count",
            )
            if key in value
        ]
        keys = preferred or list(value)
        return [item for key in keys for item in _atomic_values(value[key])]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for child in value for item in _atomic_values(child)]
    # Neo4j nodes expose their properties through ``items`` without necessarily
    # registering as Mapping.
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return _atomic_values(dict(items()))
        except Exception:
            pass
    if all(hasattr(value, coordinate) for coordinate in ("x", "y", "z")):
        return [f"POINT({value.x} {value.y} {value.z})"]
    return [value]
