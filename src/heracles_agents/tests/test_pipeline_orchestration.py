from collections import deque
from types import SimpleNamespace

from heracles_agents.llm_interface import EvalQuestion
from heracles_agents.pipelines import agentic_pipeline, canary_pipeline, feedforward_cypher_pipeline
from heracles_agents.prompt import Prompt


def make_question(uid="q1"):
    return EvalQuestion(
        uid=uid,
        name="Question",
        question="What is 1 + 1?",
        solution="2",
        correctness_comparator={"comparison_type": "SLDP", "relation": "equal"},
    )


def make_agent(tool_interface="none", template="Question: {question}"):
    return SimpleNamespace(
        agent_info=SimpleNamespace(
            tool_interface=tool_interface,
            tools={
                "tool": SimpleNamespace(to_custom=lambda: "Function name: tool")
            },
            prompt_settings=SimpleNamespace(
                base_prompt=Prompt(
                    system="sys",
                    novel_instruction_template=template,
                ),
                output_type="SLDP",
                sldp_answer_type_hint=False,
            ),
        )
    )


class FakeContext:
    instances = deque()

    def __init__(self, agent):
        self.agent = agent
        self.initial_input_tokens = 1
        self.total_output_tokens = 2
        self.n_tool_calls = 3
        self.prompt = None
        self.answer = "2"
        FakeContext.instances.append(self)

    def initialize_agent(self, prompt):
        self.prompt = prompt

    def run(self):
        return True, self.answer

    def get_agent_responses(self):
        return []


def test_canary_generate_prompt_sets_tool_and_answer_guidance():
    prompt = canary_pipeline.generate_prompt(make_question(), make_agent("custom"))

    assert prompt.novel_instruction == "Question: What is 1 + 1?"
    assert prompt.tool_description == "Function name: tool"
    assert prompt.answer_semantic_guidance == "Make your answer as concise as possible."
    assert "SLDP Equality Language" in prompt.answer_formatting_guidance


def test_agentic_generate_prompt_includes_api_prompt_for_python_dsg():
    prompt = agentic_pipeline.generate_prompt(
        make_question(),
        make_agent("none"),
        api_prompt="API docs",
    )

    rendered = prompt.to_openai_json()
    assert {"role": "developer", "content": "API docs"} in rendered


def test_canary_pipeline_happy_path(monkeypatch):
    FakeContext.instances.clear()
    monkeypatch.setattr(canary_pipeline, "AgentContext", FakeContext)
    exp = SimpleNamespace(questions=[make_question()], phases={"main": make_agent()})

    result = canary_pipeline.canary_pipeline(exp)

    analyzed = result.analyzed_questions[0]
    assert analyzed.answer == "2"
    assert analyzed.analysis.correct
    assert analyzed.analysis.valid_answer_format
    assert analyzed.analysis.input_tokens == 1
    assert analyzed.analysis.output_tokens == 2
    assert analyzed.analysis.n_tool_calls == 3
    assert analyzed.sequences[0].description == "canary-agent"
    assert FakeContext.instances[0].prompt.novel_instruction == "Question: What is 1 + 1?"


def test_agentic_pipeline_uses_python_api_prompt(monkeypatch):
    FakeContext.instances.clear()
    monkeypatch.setattr(agentic_pipeline, "AgentContext", FakeContext)
    exp = SimpleNamespace(
        questions=[make_question()],
        phases={"main": make_agent()},
        dsg_interface=SimpleNamespace(
            dsg_interface_type="python",
            get_dsg_api_prompt=lambda: "API docs",
        ),
    )

    result = agentic_pipeline.agentic_pipeline(exp)

    analyzed = result.analyzed_questions[0]
    assert analyzed.analysis.correct
    rendered = FakeContext.instances[0].prompt.to_openai_json()
    assert {"role": "developer", "content": "API docs"} in rendered


def test_feedforward_cypher_pipeline_happy_path(monkeypatch):
    FakeContext.instances.clear()

    class CypherContext(FakeContext):
        def __init__(self, agent):
            super().__init__(agent)
            self.answer = "MATCH (n) RETURN n" if len(FakeContext.instances) == 1 else "2"

    monkeypatch.setattr(feedforward_cypher_pipeline, "AgentContext", CypherContext)
    monkeypatch.setattr(
        feedforward_cypher_pipeline,
        "query_db",
        lambda dsg_interface, query: (True, "db results"),
    )
    exp = SimpleNamespace(
        questions=[make_question()],
        phases={
            "generate-cypher": make_agent(),
            "refine": make_agent(template="Question: {question}; Results: {cypher_results}"),
        },
        dsg_interface=SimpleNamespace(),
    )

    result = feedforward_cypher_pipeline.feedforward_cypher(exp)

    analyzed = result.analyzed_questions[0]
    assert analyzed.answer == "2"
    assert analyzed.analysis.correct
    assert analyzed.analysis.input_tokens == 2
    assert analyzed.analysis.output_tokens == 4
    assert analyzed.analysis.n_tool_calls == 6
    assert [sequence.description for sequence in analyzed.sequences] == [
        "cypher-producing-agent",
        "refinement-agent",
    ]
    assert "db results" in FakeContext.instances[1].prompt.novel_instruction
