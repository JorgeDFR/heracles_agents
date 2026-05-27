# ruff: noqa: F811
import logging
import time
from typing import Literal, Optional, Union

from plum import dispatch
from pydantic import BaseModel, Field

from heracles_agents.agent_functions import (
    call_function,
    count_message_tokens,
    extract_answer,
    extract_answer_tag,
    get_answer_tool_payload,
    generate_prompt_for_agent,
    generate_update_for_history,
    get_text_body,
    is_answer_tool_call,
    is_custom_tool_call,
    is_function_call,
    iterate_messages,
    make_tool_response,
    normalize_message,
)
from heracles_agents.exceptions import (
    LlmRateLimitError,
    LlmTimeoutError,
    LlmServiceUnavailableError,
    LlmConnectionError,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.normalized_response import (
    NormalizedMessage,
    normalized_summary,
)
from heracles_agents.tool_rendering import render_tool_for_interface

logger = logging.getLogger(__name__)


class SldpComparison(BaseModel):
    comparison_type: Literal["SLDP"]
    relation: str  # equal, subset, superset


class PddlComparison(BaseModel):
    comparison_type: Literal["PDDL"]
    relation: str  # equal, subset, superset


class LLmJudgeComparison(BaseModel):
    comparison_type: Literal["LLM_JUDGE"]


ComparisonType = Union[SldpComparison, PddlComparison, LLmJudgeComparison]


class EvalQuestion(BaseModel):
    name: str
    question: str
    solution: str
    uid: str | int
    tags: Optional[list[str]] = None
    correctness_comparator: ComparisonType = Field(discriminator="comparison_type")


class AgentResponse(BaseModel):
    # "full" response from the LLM (what format?)
    raw_response: str
    # interpretable response from the LLM (e.g., tool call, parsed final answer)
    parsed_response: Optional[str]
    analysis: "ResponseAnalysis" = Field(default_factory=lambda: ResponseAnalysis())


class ResponseAnalysis(BaseModel):
    valid_sldp: bool = False
    valid_cypher: bool = False
    tool_call_succeeded: Optional[bool] = None


class AgentSequence(BaseModel):
    # What the "purpose" of this agent sequence is. Eventually should be more
    # structured/dispatchable than string?
    description: str

    responses: list[AgentResponse]


class QuestionAnalysis(BaseModel):
    # Information that is relevant about evaluating the response quality of the
    # "whole question"

    valid_answer_format: bool
    correct: bool
    input_tokens: int
    output_tokens: int
    n_tool_calls: int


class AnalyzedQuestion(BaseModel):
    question: EvalQuestion
    sequences: list[AgentSequence]
    answer: Optional[str]
    analysis: Optional[QuestionAnalysis]
    completed: bool = True


class AnalyzedQuestions(BaseModel):
    analyzed_questions: list[AnalyzedQuestion]


class AnalyzedExperiment(BaseModel):
    experiment_configurations: dict[str, AnalyzedQuestions]
    metadata: dict = Field(default_factory=dict)


def generate_tools_for_agent(agent_info):
    match agent_info.tool_interface:
        case "custom":
            explicit_tools = []
        case "none":
            explicit_tools = []
        case "openai" | "anthropic" | "ollama" | "bedrock" | "openrouter":
            explicit_tools = [
                render_tool_for_interface(tool, agent_info.tool_interface)
                for tool in agent_info.tools.values()
            ]
        case _:
            raise NotImplementedError(
                f"Unknown tool interface: {agent_info.tool_interface}"
            )
    return explicit_tools


def process_answer(agent: LlmAgent, message):
    match agent.agent_info.prompt_settings.output_type:
        case "SLDP_TOOL" | "PDDL_TOOL":
            if is_answer_tool_call(agent, message):
                return get_answer_tool_payload(agent, message)

        # case "SLDP" | "PDDL":
        case _:
            answer = extract_answer(agent, extract_answer_tag, message)
            return answer


def needs_tool_processing(agent: LlmAgent, message):
    if is_answer_tool_call(agent, message):
        return False
    return is_function_call(agent, message) or is_custom_tool_call(agent, message)


@dispatch
def get_summary_text(resp: str):
    return resp


@dispatch
def get_summary_text(resp: type(None)):
    return ""


@dispatch
def get_summary_text(resp: list):
    return "\n".join(get_summary_text(r) for r in resp)


@dispatch
def get_summary_text(resp):
    return normalized_summary(normalize_message(None, resp))


@dispatch
def get_summary_text(resp: NormalizedMessage):
    return normalized_summary(resp)


def make_agent_response(resp):
    return AgentResponse(raw_response=str(resp), parsed_response=get_summary_text(resp))


@dispatch
def get_summary_text(resp: dict):
    if "text" in resp:
        return resp["text"]
    elif "role" in resp and "content" in resp:
        if isinstance(resp["content"], list):
            return (
                resp["role"]
                + ":"
                + "\n".join(get_summary_text(r) for r in resp["content"])
            )
        else:
            return resp["role"] + ": " + resp["content"]
    elif "type" in resp and resp["type"] == "function_call_output" and "output" in resp:
        return "Function result: " + str(resp["output"])
    elif "toolResult" in resp:
        try:
            return "Tool result: " + "\n".join(
                r["text"] for r in resp["toolResult"]["content"]
            )
        except Exception as ex:
            logger.error("bedrock logging format error")
            logger.error(str(ex))
            return "Tool result: " + str(resp["toolResult"])
    else:
        logger.warning(f"Don't know how to turn dict {resp} into elegant string")
        return str(resp)


class AgentContext:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.history = []
        self.n_tool_calls = 0
        self.initial_input_tokens = 0
        self.total_output_tokens = 0

    def initialize_agent(self, prompt):
        self.history = generate_prompt_for_agent(prompt, self.agent)
        self.initial_input_tokens = count_message_tokens(self.agent, self.history)

        logger.info(f"Agent inintialized with: \n{get_summary_text(self.history)}")

    def call_llm(self, history):
        model_info = self.agent.model_info
        logger.debug(f"Calling llm with history: {history}")

        explicit_tools = generate_tools_for_agent(self.agent.agent_info)

        response_format = getattr(model_info, "response_format", "text")

        n_retries = 5
        wait_time_s = 60
        last_exception = None
        for idx in range(n_retries):
            try:
                return self.agent.client.call(
                    model_info,
                    explicit_tools,
                    response_format,
                    history,
                )

            # ----------------------------------------------------------
            # Retryable errors
            # ----------------------------------------------------------
            except (
                LlmRateLimitError,
                LlmTimeoutError,
                LlmServiceUnavailableError,
                LlmConnectionError,
            ) as ex:
                last_exception = ex
                logging.warning(
                    f"{type(ex).__name__}: {ex}\n"
                    f"Retrying in {wait_time_s}s "
                    f"({idx + 1}/{n_retries})"
                )
                time.sleep(wait_time_s)

            # ----------------------------------------------------------
            # Fatal provider errors
            # ----------------------------------------------------------
            except Exception as ex:
                logger.error(f"LLM provider call failed: {ex}")
                raise

        raise RuntimeError(f"LLM call failed after {n_retries} retries") from last_exception

    def handle_response(self, response):
        executed_tool_calls = []
        logger.debug(f"Handling response: {response}")
        for message in iterate_messages(self.agent, response):
            output_tokens = count_message_tokens(self.agent, message)
            self.total_output_tokens += output_tokens
            message_text = normalize_message(self.agent, message).text
            if message_text is not None:
                logger.debug(f"Processing message ({type(message)}: {message_text}")
            if not needs_tool_processing(self.agent, message):
                continue

            self.n_tool_calls += 1

            result = call_function(self.agent, message)
            logger.debug(f"function_result: {result}")
            tool_response = make_tool_response(self.agent, message, result)
            logger.debug(f"Tool response: {result}")
            executed_tool_calls.append(tool_response)

        return executed_tool_calls

    def update_history(self, response):
        logger.info(f"History update: \n{get_summary_text(response)}")
        update = generate_update_for_history(self.agent, response)
        self.history += update

    def check_if_done(self, history, response, last_update):
        # If any of the LLM's response messages called an answer tool, we are done
        for message in iterate_messages(self.agent, response):
            if is_answer_tool_call(self.agent, message):
                return True
        # If the LLM didn't call an answer tool, we are done if the LLM didn't call *any* tools.
        return len(last_update) == 0

    def step(self):
        logger.debug("Agent stepping")
        try:
            response = self.call_llm(self.history)
            logger.debug(f"Got response: {response}")

        except Exception:
            response = {
                "role": "assistant",
                "content": "Unable to contact LLM provider server!",
            }
            self.history.append(response)
            return False, False

        update = self.handle_response(response)
        logger.debug(f"Tool update: {update}")

        self.update_history(response)
        self.update_history(update)
        done = self.check_if_done(self.history, response, update)
        return True, done

    def get_agent_responses(self):
        return [make_agent_response(resp) for resp in self.history]

    def run(self):
        for i in range(self.agent.agent_info.max_iterations):
            success, done = self.step()
            if done or not success:
                break
        if success and done:
            answer = process_answer(self.agent, self.history[-1])
        else:
            answer = None
        logger.info(f"Agent exiting. Finished before max iteration cap? {done}")
        logger.info(f"Agent used {self.n_tool_calls} tool calls")
        logger.info(f"Answer: {answer}")
        return done, answer
