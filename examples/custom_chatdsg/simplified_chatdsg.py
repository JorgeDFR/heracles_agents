#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import load_dsg_to_db

from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def new_user_message(text):
    return [{"role": "user", "content": text}]


def generate_initial_prompt(agent: LlmAgent):
    prompt = agent.agent_info.prompt_settings.base_prompt
    return prompt


def main():
    parser = argparse.ArgumentParser("ChatDSG agent (terminal mode)")
    parser.add_argument(
        "--scene-graph",
        nargs="?",
        const=None,
        default=None,
        help="DSG Filepath to load",
    )
    parser.add_argument(
        "--object-labelspace",
        type=str,
        help="Path to object labelspace",
        default="ade20k_mit_label_space.yaml",
    )
    parser.add_argument(
        "--room-labelspace",
        type=str,
        help="Path to room labelspace",
        default="b45_label_space.yaml",
    )
    parser.add_argument("--db_ip", type=str, help="Heracles database ip")
    parser.add_argument("--db_port", type=int, help="Heracles database port")
    args = parser.parse_args()

    if args.db_ip is None:
        args.db_ip = os.getenv("ADT4_HERACLES_IP")

    if args.db_port is None:
        args.db_port = os.getenv("ADT4_HERACLES_PORT")

    # Optional DSG loading
    if args.scene_graph:
        dsg_filepath = args.scene_graph
        logger.info(f"Loading DSG into database from filepath: {dsg_filepath}")

        scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
        summarize_dsg(scene_graph)

        neo4j_uri = f"neo4j://{args.db_ip}:{args.db_port}"
        neo4j_creds = (
            os.getenv("HERACLES_NEO4J_USERNAME"),
            os.getenv("HERACLES_NEO4J_PASSWORD"),
        )

        load_dsg_to_db(
            args.object_labelspace,
            args.room_labelspace,
            neo4j_uri,
            neo4j_creds,
            scene_graph,
        )

        logger.info("DSG loaded!")

    # Load agent config
    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)

    agent = LlmAgent(**yml)

    # Initialize conversation
    messages = generate_initial_prompt(agent).to_openai_json(
        "Now you will interact with the user:"
    )

    print("\n=== ChatDSG Agent (Terminal Mode) ===")
    print("Type 'exit' or 'quit' to stop.\n")

    # Main input loop
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        if not user_input:
            continue

        # Append user message
        messages += new_user_message(user_input)
        initial_length = len(messages)

        # Run agent
        cxt = AgentContext(agent)
        cxt.history = messages

        success, answer = cxt.run()

        if not success:
            print("Agent failed to produce a response.")
            continue

        responses = cxt.get_agent_responses()

        # Print only new responses
        for r in responses[initial_length:]:
            print("\nAgent:")
            print(r.parsed_response)
            print()

        # Update history with agent responses
        messages = cxt.history


if __name__ == "__main__":
    main()