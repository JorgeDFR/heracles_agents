import copy
import logging
from time import perf_counter

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
    make_usage_token_metrics,
    make_latency_metrics,
)
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.pipelines.in_context_utils import scene_graph_to_prompt_full
from heracles_agents.pipelines.local_metrics import (
    make_local_resource_metrics,
    make_question_local_resource_metrics,
    prepare_local_resource_monitor,
    start_local_resource_measurement,
)
from heracles_agents.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


def generate_prompt(
    incontext_dsg_interface,
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {},
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)

    dsg_desciption = scene_graph_to_prompt_full(
        incontext_dsg_interface.get_dsg(),
        incontext_dsg_interface.get_place_layer_name(),
    )
    try:
        prompt.novel_instruction = prompt.novel_instruction_template.format(
            question=question.question, dsg_description=dsg_desciption
        )
    except KeyError as ex:
        logger.error("Novel instruction template has unfilled parameter!")
        raise ex

    prompt.answer_semantic_guidance = "Make your answer as concise as possible."
    prompt.answer_formatting_guidance = get_answer_formatting_guidance(
        agent_config, question
    )

    return prompt


def incontext_dsg(exp):
    analyzed_questions = []
    local_monitor = prepare_local_resource_monitor(exp)
    local_measurement = start_local_resource_measurement(local_monitor)
    all_contexts = []
    for question in exp.questions:
        question_started = perf_counter()
        question_resource_started = local_measurement.timestamp()
        contexts = []
        parsing_validation_seconds = 0.0
        answer = None
        sequences = []
        completed = False
        try:
            cxt = AgentContext(exp.phases["main"])
            contexts.append(cxt)

            prompt = generate_prompt(exp.dsg_interface, question, exp.phases["main"])

            cxt.initialize_agent(prompt)
            success, answer = cxt.run()
            logger.debug(f"\nLLM Final Answer: {answer}\n")

            sequence = AgentSequence(
                description="in-context pipeline", responses=cxt.get_agent_responses()
            )
            sequences.append(sequence)

            validation_started = perf_counter()
            try:
                valid_format, correct = evaluate_answer(
                    question.correctness_comparator, answer, question.solution
                )
            finally:
                parsing_validation_seconds += perf_counter() - validation_started

            logger.debug(f"\n\nCorrect? {correct}\n\n")

            analysis = QuestionAnalysis(
                correct=correct,
                valid_answer_format=valid_format,
                **make_usage_token_metrics(contexts),
                output_tokens=cxt.total_output_tokens,
                n_tool_calls=cxt.n_tool_calls,
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
                **make_usage_token_metrics(contexts),
                output_tokens=0,
                n_tool_calls=0,
                latency=make_latency_metrics(
                    contexts,
                    end_to_end_seconds=perf_counter() - question_started,
                    parsing_validation_seconds=parsing_validation_seconds,
                ),
                cost=make_cost_metrics(contexts),
            )

        analysis.local_resources = make_question_local_resource_metrics(
            local_measurement, question_resource_started, local_measurement.timestamp()
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


main_phase = PipelinePhase(
    name="main",
    description="Map question to answer using in-context scene graph",
)

d = PipelineDescription(
    name="feedforward_in_context_full",
    description="in-context scene graph",
    phases=[main_phase],
    function=incontext_dsg,
)

register_pipeline(d)

if __name__ == "__main__":
    import yaml

    from heracles_agents.experiment_definition import ExperimentConfiguration
    from heracles_agents.cli.summarize_results import display_experiment_results

    logging.basicConfig(level=logging.INFO)

    with open("experiments/incontext_full_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    experiment = ExperimentConfiguration(**yml)
    logger.debug(f"Loaded experiment configuration: {experiment}")

    aqs = incontext_dsg(experiment)
    with open("output/feedforward_incontext_full_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
