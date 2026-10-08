import pytest

from heracles_agents.cli import question_validator
from heracles_agents.llm_interface import EvalQuestion


def make_question(**overrides):
    data = {
        "uid": "q1",
        "name": "Example",
        "question": "What is true?",
        "solution": "2",
        "tags": [],
        "correctness_comparator": {
            "comparison_type": "SLDP",
            "relation": "equal",
        },
    }
    data.update(overrides)
    return EvalQuestion(**data)


def test_load_yaml_builds_eval_questions(tmp_path):
    questions_file = tmp_path / "questions.yaml"
    questions_file.write_text(
        """
questions:
  - uid: q1
    name: Example
    question: What is true?
    solution: "2"
    tags: [smoke]
    correctness_comparator:
      comparison_type: SLDP
      relation: equal
""",
        encoding="utf-8",
    )

    questions = question_validator.load_yaml(questions_file)

    assert len(questions) == 1
    assert questions[0].name == "Example"
    assert questions[0].tags == ["smoke"]


def test_validate_solution_helpers_return_boolean():
    assert question_validator.validate_sldp_solution("2")
    assert not question_validator.validate_sldp_solution("(")

    assert question_validator.validate_pddl_solution("(and)")
    assert not question_validator.validate_pddl_solution("(")


def test_render_table_handles_tags_and_validation(monkeypatch):
    printed = []
    monkeypatch.setattr(question_validator.console, "print", printed.append)

    question_validator.render_table(
        [make_question(tags=[]), make_question(uid="q2", tags=["a", "b"])],
        show_solutions=True,
        show_tags=True,
        validate=True,
    )

    assert len(printed) == 1
    table = printed[0]
    assert [column.header for column in table.columns] == [
        "UID",
        "Name",
        "Question",
        "Solution",
        "Tags",
        "Valid?",
    ]


def test_render_table_rejects_unknown_comparator(monkeypatch):
    printed = []
    monkeypatch.setattr(question_validator.console, "print", printed.append)
    question = make_question()
    question.correctness_comparator.comparison_type = "UNKNOWN"

    with pytest.raises(Exception, match="Unknown comparison type UNKNOWN"):
        question_validator.render_table([question], validate=True)


def test_question_overlap_metadata_is_preserved_and_validated():
    question = make_question(question_type="qa_object_total", overlap_class="direct")
    serialized = question.model_dump(mode="json")
    assert serialized["question_type"] == "qa_object_total"
    assert serialized["overlap_class"] == "direct"
    assert EvalQuestion.model_validate(serialized).overlap_class == "direct"
    assert make_question().overlap_class is None
    with pytest.raises(ValueError, match="overlap_class"):
        make_question(overlap_class="unknown")
