import copy
import logging

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
    QuestionAnalysis,
    EvalQuestion,
    LlmAgent,
    QuestionAnalysis,
)
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


def generate_prompt(
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {},
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)
    if agent_config.agent_info.tool_interface == "custom":
        prompt.tool_description = "\n".join(
            [t.to_custom() for t in agent_config.agent_info.tools.values()]
        )

    try:
        prompt.novel_instruction = prompt.novel_instruction_template.format(
            question=question.question, **task_state_context
        )
    except KeyError as ex:
        logger.error("Novel instruction template has unfilled parameter!")
        print(ex)
        raise ex

    prompt.answer_semantic_guidance = "Make your answer as concise as possible"
    prompt.answer_formatting_guidance = get_answer_formatting_guidance(
        agent_config, question
    )

    return prompt


def canary_pipeline(exp):
    analyzed_questions = []

    for question in exp.questions:
        try:
            logger.info(f"\n=======================\nQuestion: {question.question}\n")
            cxt = AgentContext(exp.phases["main"])

            prompt = generate_prompt(question, exp.phases["main"])
            logger.info(f"\nLLM Prompt: {prompt}\n")

            cxt.initialize_agent(prompt)
            success, answer = cxt.run()
            logger.info(f"\nLLM Answer: {answer}\n")

            valid_format, correct = evaluate_answer(
                question.correctness_comparator, answer, question.solution
            )
            logger.info(f"\n\nCorrect? {correct}\n\n")

            agent_sequence = AgentSequence(
                description="tool-calling-agent", responses=cxt.get_agent_responses()
            )

            analysis = QuestionAnalysis(
                correct=correct,
                valid_answer_format=valid_format,
                input_tokens=cxt.initial_input_tokens,
                output_tokens=cxt.total_output_tokens,
                n_tool_calls=cxt.n_tool_calls,
            )

        except Exception as ex:
            print(ex)
            logger.error("Bad Question!")
            logger.error(str(ex))
            analysis = QuestionAnalysis(
                correct=False,
                valid_answer_format=False,
                input_tokens=0,
                output_tokens=0,
                n_tool_calls=0,
            )

        aq = AnalyzedQuestion(
            question=question,
            answer=answer,
            sequences=[agent_sequence],
            analysis=analysis,
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(
    name="main",
    description="Main canary phase"
)

d = PipelineDescription(
    name="canary",
    description="For initial testing",
    phases=[main_phase],
    function=canary_pipeline,
)

register_pipeline(d)
