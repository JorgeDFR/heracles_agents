import copy
import logging
from time import perf_counter

from heracles_agents.agent_functions import build_custom_tool_prompt
from heracles_agents.experiment_definition import (
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
)
from heracles_agents.llm_interface import (
    AgentContext,
    AgentSequence,
    AnalyzedQuestion,
    AnalyzedQuestions,
    EvalQuestion,
    LlmAgent,
    QuestionAnalysis,
    make_cost_metrics,
    make_input_token_metrics,
    make_latency_metrics,
)
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.pipelines.db_utils import query_db
from heracles_agents.pipelines.local_metrics import (
    make_local_resource_metrics,
    prepare_local_resource_monitor,
    start_local_resource_measurement,
)
from heracles_agents.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


def generate_prompt(
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {}
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)
    if agent_config.agent_info.tool_interface == "custom":
        prompt.tool_description = build_custom_tool_prompt(
            agent_config.agent_info.tools.values()
        )

    try:
        prompt.novel_instruction = prompt.novel_instruction_template.format(
            question=question.question, **task_state_context
        )
    except KeyError as ex:
        logger.error("Novel instruction template has unfilled parameter!")
        raise ex

    prompt.answer_semantic_guidance = "Make your answer as concise as possible."
    formatting = get_answer_formatting_guidance(agent_config, question)
    if formatting is not None:
        prompt.answer_formatting_guidance = formatting

    return prompt


def feedforward_cypher(exp):
    analyzed_questions = []
    local_monitor = prepare_local_resource_monitor(exp)
    local_measurement = start_local_resource_measurement(local_monitor)
    all_contexts = []

    for question in exp.questions:
        question_started = perf_counter()
        contexts = []
        neo4j_query_seconds = 0.0
        parsing_validation_seconds = 0.0
        answer = None
        sequences = []
        completed = False
        try:
            logger.debug(f"\n=======================\nQuestion: {question.question}\n")
            cxt = AgentContext(exp.phases["generate-cypher"])
            contexts.append(cxt)

            prompt = generate_prompt(question, exp.phases["generate-cypher"])
            #logger.debug(f"\nLLM Prompt (Generate Cypher): {prompt}\n")

            cxt.initialize_agent(prompt)
            success, answer = cxt.run()
            logger.debug(f"\nLLM Intermediate Answer: {answer}\n")

            cypher_generation_sequence = AgentSequence(
                description="cypher-producing-agent",
                responses=cxt.get_agent_responses(),
            )
            sequences.append(cypher_generation_sequence)

            query_started = perf_counter()
            try:
                success, query_result = query_db(exp.dsg_interface, answer)
            finally:
                neo4j_query_seconds += perf_counter() - query_started

            cxt2 = AgentContext(exp.phases["refine"])
            contexts.append(cxt2)
            refinement_prompt = generate_prompt(
                question,
                exp.phases["refine"],
                {"cypher_results": query_result, "cypher_query": answer},
            )
            #logger.debug(f"\nLLM Prompt (Refine): {refinement_prompt}\n")

            cxt2.initialize_agent(refinement_prompt)
            success, answer = cxt2.run()
            logger.debug(f"LLM Final Answer: {answer}")

            validation_started = perf_counter()
            try:
                valid_format, correct = evaluate_answer(
                    question.correctness_comparator, answer, question.solution
                )
            finally:
                parsing_validation_seconds += perf_counter() - validation_started

            logger.debug(f"\n\nCorrect? {correct}\n\n")

            refinement_sequence = AgentSequence(
                description="refinement-agent", responses=cxt2.get_agent_responses()
            )
            sequences.append(refinement_sequence)

            n_output_tokens = cxt.total_output_tokens + cxt2.total_output_tokens
            n_tool_calls = cxt.n_tool_calls + cxt2.n_tool_calls

            analysis = QuestionAnalysis(
                correct=correct,
                valid_answer_format=valid_format,
                **make_input_token_metrics(contexts),
                output_tokens=n_output_tokens,
                n_tool_calls=n_tool_calls,
                latency=make_latency_metrics(
                    contexts,
                    end_to_end_seconds=perf_counter() - question_started,
                    parsing_validation_seconds=parsing_validation_seconds,
                    neo4j_query_seconds=neo4j_query_seconds,
                ),
                cost=make_cost_metrics(contexts),
            )
            completed = True

        except Exception as ex:
            logger.error("Bad Question!")
            logger.error(str(ex))
            analysis = QuestionAnalysis(
                correct=False,
                valid_answer_format=False,
                **make_input_token_metrics(contexts),
                output_tokens=0,
                n_tool_calls=0,
                latency=make_latency_metrics(
                    contexts,
                    end_to_end_seconds=perf_counter() - question_started,
                    parsing_validation_seconds=parsing_validation_seconds,
                    neo4j_query_seconds=neo4j_query_seconds,
                ),
                cost=make_cost_metrics(contexts),
            )

        all_contexts.extend(contexts)
        aq = AnalyzedQuestion(
            question=question,
            answer=answer,
            sequences=sequences,
            analysis=analysis,
            completed=completed,
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(
        analyzed_questions=analyzed_questions,
        local_resources=make_local_resource_metrics(all_contexts, local_measurement),
    )
    return aqs


cypher_phase = PipelinePhase(
    name="generate-cypher", description="Map question to Cypher query"
)

refine_phase = PipelinePhase(
    name="refine", description="Map result of cypher query to final answer"
)

d = PipelineDescription(
    name="feedforward_cypher",
    description="Single cypher query, then refinement",
    phases=[cypher_phase, refine_phase],
    function=feedforward_cypher,
)

register_pipeline(d)
