import copy
import logging
import os
from time import perf_counter

from heracles_agents.experiment_definition import (
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
)
from heracles_agents.agent_functions import build_custom_tool_prompt
from heracles_agents.llm_interface import (
    AgentContext,
    AgentSequence,
    AnalyzedQuestion,
    AnalyzedQuestions,
    EvalQuestion,
    LlmAgent,
    QuestionAnalysis,
    make_cost_metrics,
    make_latency_metrics,
)
from heracles_agents.pipelines.codegen_utils import (
    execute_generated_code,
    load_dsg,
)
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


def generate_prompt(
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {},
    api_prompt: str = None,
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)

    if api_prompt:
        prompt.set_api_prompt(api_prompt)

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


def feedforward_codegen(exp):
    analyzed_questions = []
    dsg_filepath = os.path.expandvars(exp.dsg_interface.dsg_filepath)
    dsg_labels_filepath = (
        os.path.expandvars(exp.dsg_interface.dsg_labels_filepath)
        if hasattr(exp.dsg_interface, "dsg_labels_filepath")
        and exp.dsg_interface.dsg_labels_filepath
        else None
    )
    scene_graph = load_dsg(dsg_filepath, dsg_labels_filepath)
    copy_dsg_per_question = getattr(exp.dsg_interface, "copy_dsg_per_question", False)
    # Set api in prompt
    api_string = exp.dsg_interface.get_dsg_api_prompt()
    for question in exp.questions:
        question_started = perf_counter()
        contexts = []
        parsing_validation_seconds = 0.0
        answer = None
        sequences = []
        completed = False
        try:
            logger.debug(f"\n=======================\nQuestion: {question.question}\n")
            question_scene_graph = (
                copy.deepcopy(scene_graph) if copy_dsg_per_question else scene_graph
            )
            cxt = AgentContext(exp.phases["generate-code"])
            contexts.append(cxt)

            prompt = generate_prompt(
                question, exp.phases["generate-code"], api_prompt=api_string
            )

            cxt.initialize_agent(prompt)
            success, answer = cxt.run()
            if not success:
                raise RuntimeError("Code generation agent failed before producing code")
            logger.debug(f"\nLLM Intermediate Answer: {answer}\n")

            codgen_sequence = AgentSequence(
                description="codegen-agent", responses=cxt.get_agent_responses()
            )
            sequences.append(codgen_sequence)

            success, code_results = execute_generated_code(answer, question_scene_graph)

            cxt2 = AgentContext(exp.phases["refine"])
            contexts.append(cxt2)
            refinement_prompt = generate_prompt(
                question,
                exp.phases["refine"],
                {"python_results": code_results, "python_code": answer},
            )

            cxt2.initialize_agent(refinement_prompt)
            success, answer = cxt2.run()
            if not success:
                raise RuntimeError("Refinement agent failed before producing an answer")
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

            n_input_tokens = cxt.initial_input_tokens + cxt2.initial_input_tokens
            n_output_tokens = cxt.total_output_tokens + cxt2.total_output_tokens

            analysis = QuestionAnalysis(
                correct=correct,
                valid_answer_format=valid_format,
                input_tokens=n_input_tokens,
                output_tokens=n_output_tokens,
                n_tool_calls=cxt.n_tool_calls + cxt2.n_tool_calls,
                latency=make_latency_metrics(
                    contexts,
                    end_to_end_seconds=perf_counter() - question_started,
                    parsing_validation_seconds=parsing_validation_seconds,
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
                input_tokens=0,
                output_tokens=0,
                n_tool_calls=0,
                latency=make_latency_metrics(
                    contexts,
                    end_to_end_seconds=perf_counter() - question_started,
                    parsing_validation_seconds=parsing_validation_seconds,
                ),
                cost=make_cost_metrics(contexts),
            )

        aq = AnalyzedQuestion(
            question=question,
            answer=answer,
            sequences=sequences,
            analysis=analysis,
            completed=completed,
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


codegen_phase = PipelinePhase(
    name="generate-code", description="Map question to Python code"
)
refine_phase = PipelinePhase(
    name="refine", description="Map result of executed code to final answer"
)
d = PipelineDescription(
    name="feedforward_codegen",
    description="Single codegen query, then refinement",
    phases=[codegen_phase, refine_phase],
    function=feedforward_codegen,
)

register_pipeline(d)
