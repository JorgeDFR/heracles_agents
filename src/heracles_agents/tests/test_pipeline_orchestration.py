from collections import deque
from types import SimpleNamespace

from heracles_agents.llm_interface import AgentResponse, EvalQuestion
from heracles_agents.pipelines import (
    agentic_pipeline,
    canary_pipeline,
    feedforward_codegen_pipeline,
    feedforward_cypher_pipeline,
)
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.openai.prompt_rendering import (
    render_openai_prompt,
)
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription


def make_question(uid="q1"):
    return EvalQuestion(
        uid=uid,
        name="Question",
        question="What is 1 + 1?",
        solution="2",
        correctness_comparator={"comparison_type": "SLDP", "relation": "equal"},
    )


def make_agent(tool_interface="none", template="Question: {question}"):
    def tool_fn(value: str):
        return value

    return SimpleNamespace(
        agent_info=SimpleNamespace(
            tool_interface=tool_interface,
            tools={
                "tool": ToolDescription(
                    name="tool",
                    description="Tool",
                    parameters=[FunctionParameter("value", str, "Value")],
                    function=tool_fn,
                )
            },
            prompt_settings=SimpleNamespace(
                base_prompt=Prompt(
                    system="sys",
                    novel_instruction_template=template,
                ),
                output_type="SLDP",
                answer_type_hint=False,
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
        self.llm_call_seconds = 0.1
        self.tool_execution_seconds = 0.2
        self.retry_wait_seconds = 0.3
        self.time_to_first_token_seconds = None
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
    assert "The following tools can be used" in prompt.tool_description
    assert "Function name: tool" in prompt.tool_description
    assert prompt.answer_semantic_guidance == "Make your answer as concise as possible."
    assert "SLDP Equality Language" in prompt.answer_formatting_guidance


def test_agentic_generate_prompt_includes_api_prompt_for_python_dsg():
    prompt = agentic_pipeline.generate_prompt(
        make_question(),
        make_agent("none"),
        api_prompt="API docs",
    )

    rendered = render_openai_prompt(prompt)
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
    assert analyzed.analysis.latency.end_to_end_seconds >= 0
    assert analyzed.analysis.latency.llm_call_seconds == 0.1
    assert analyzed.analysis.latency.tool_execution_seconds == 0.2
    assert analyzed.analysis.latency.retry_wait_seconds == 0.3
    assert analyzed.analysis.local_resources is None
    assert analyzed.sequences[0].description == "canary-agent"
    assert FakeContext.instances[0].prompt.novel_instruction == "Question: What is 1 + 1?"


def test_canary_pipeline_marks_failed_question_incomplete(monkeypatch):
    class FailingContext(FakeContext):
        def run(self):
            raise RuntimeError("provider down")

    FakeContext.instances.clear()
    monkeypatch.setattr(canary_pipeline, "AgentContext", FailingContext)
    exp = SimpleNamespace(questions=[make_question()], phases={"main": make_agent()})

    result = canary_pipeline.canary_pipeline(exp)

    analyzed = result.analyzed_questions[0]
    assert analyzed.answer is None
    assert analyzed.sequences == []
    assert analyzed.completed is False
    assert analyzed.analysis.correct is False
    assert analyzed.analysis.latency.end_to_end_seconds >= 0


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
    rendered = render_openai_prompt(FakeContext.instances[0].prompt)
    assert {"role": "developer", "content": "API docs"} in rendered


def test_agentic_pipeline_preserves_failed_context_stats_and_sequence(monkeypatch):
    class FailingContext(FakeContext):
        def run(self):
            self.initial_input_tokens = 11
            self.total_output_tokens = 7
            self.n_tool_calls = 1
            self.history = ["prompt", "assistant tool call"]
            raise RuntimeError("tool failed")

        def get_agent_responses(self):
            return [
                AgentResponse(
                    raw_response="assistant tool call",
                    parsed_response="assistant tool call",
                )
            ]

    FakeContext.instances.clear()
    monkeypatch.setattr(agentic_pipeline, "AgentContext", FailingContext)
    exp = SimpleNamespace(
        questions=[make_question()],
        phases={"main": make_agent()},
        dsg_interface=SimpleNamespace(dsg_interface_type="none"),
    )

    result = agentic_pipeline.agentic_pipeline(exp)

    analyzed = result.analyzed_questions[0]
    assert analyzed.answer is None
    assert analyzed.completed is False
    assert analyzed.analysis.input_tokens == 11
    assert analyzed.analysis.output_tokens == 7
    assert analyzed.analysis.n_tool_calls == 1
    assert analyzed.sequences[0].description == "cypher-agent-failed-1"
    assert analyzed.sequences[0].responses[0].parsed_response == "assistant tool call"


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
    assert analyzed.analysis.latency.llm_call_seconds == 0.2
    assert analyzed.analysis.latency.tool_execution_seconds == 0.4
    assert analyzed.analysis.latency.neo4j_query_seconds >= 0
    assert [sequence.description for sequence in analyzed.sequences] == [
        "cypher-producing-agent",
        "refinement-agent",
    ]
    assert "db results" in FakeContext.instances[1].prompt.novel_instruction


def test_feedforward_codegen_pipeline_happy_path(monkeypatch):
    FakeContext.instances.clear()

    class CodegenContext(FakeContext):
        def __init__(self, agent):
            super().__init__(agent)
            self.answer = "def solve_task(G): return 2" if len(FakeContext.instances) == 1 else "2"

    monkeypatch.setattr(feedforward_codegen_pipeline, "AgentContext", CodegenContext)
    monkeypatch.setattr(feedforward_codegen_pipeline, "load_dsg", lambda path, labels: ["graph"])
    monkeypatch.setattr(
        feedforward_codegen_pipeline,
        "execute_generated_code",
        lambda code, graph: (True, "2"),
    )
    exp = SimpleNamespace(
        questions=[make_question()],
        phases={
            "generate-code": make_agent(),
            "refine": make_agent(
                template="Question: {question}; Code: {python_code}; Results: {python_results}"
            ),
        },
        dsg_interface=SimpleNamespace(
            dsg_filepath="$HOME/graph.dsg",
            dsg_labels_filepath=None,
            copy_dsg_per_question=True,
            get_dsg_api_prompt=lambda: "API docs",
        ),
    )

    result = feedforward_codegen_pipeline.feedforward_codegen(exp)

    analyzed = result.analyzed_questions[0]
    assert analyzed.answer == "2"
    assert analyzed.analysis.correct
    assert analyzed.analysis.input_tokens == 2
    assert analyzed.analysis.output_tokens == 4
    assert analyzed.analysis.n_tool_calls == 6
    assert analyzed.analysis.latency.llm_call_seconds == 0.2
    assert analyzed.analysis.latency.tool_execution_seconds == 0.4
    assert [sequence.description for sequence in analyzed.sequences] == [
        "codegen-agent",
        "refinement-agent",
    ]
    assert "API docs" in [
        item["content"] for item in render_openai_prompt(FakeContext.instances[0].prompt)
    ]
    assert "Results: 2" in FakeContext.instances[1].prompt.novel_instruction
