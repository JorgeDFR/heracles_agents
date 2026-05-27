#!/usr/bin/env python3

import os
import io
import json
import copy
import yaml
import logging
import argparse
import contextlib

import spark_dsg

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from heracles.dsg_utils import summarize_dsg
from heracles.utils import load_dsg_to_db

from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Rich console
# ------------------------------------------------------------------------------

console = Console()

# ------------------------------------------------------------------------------
# Runtime flags
# ------------------------------------------------------------------------------

DEBUG_INTERNAL = False
SAVE_HISTORY = False

HISTORY_FILE = "chat_history.json"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def new_user_message(text):
    return [{"role": "user", "content": text}]


def generate_initial_prompt(agent_config: LlmAgent):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)

    if agent_config.agent_info.tool_interface == "custom":
        prompt.tool_description = "\n".join(
            [t.to_custom() for t in agent_config.agent_info.tools.values()]
        )

    return prompt


def is_tool_message(text: str) -> bool:
    """
    Detect raw tool messages.
    """

    if not text:
        return False

    lowered = text.lower().strip()

    return (
        lowered.startswith("tool:")
        or lowered.startswith("function:")
        or lowered.startswith("observation:")
    )


def print_internal_message(text: str):
    """
    Pretty print internal/tool/debug messages.
    """

    console.print(
        Panel(
            Markdown(f"```text\n{text}\n```"),
            title="Internal Debug",
            border_style="yellow",
        )
    )


def extract_final_response(responses):
    """
    Extract only the final clean assistant response.
    """

    final_response = None

    for r in responses:
        text = getattr(r, "parsed_response", None)

        # Skip empty messages
        if text is None:
            continue

        text = str(text).strip()

        if not text:
            continue

        # Skip internal/tool messages
        if is_tool_message(text):
            continue

        # --------------------------------------------------------------
        # Normal assistant response
        # --------------------------------------------------------------

        final_response = text

    return final_response


def save_history(messages):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(messages, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to save history: {e}")


def print_welcome():
    mode = (
        "[bold yellow]DEBUG MODE ENABLED[/bold yellow]\n"
        if DEBUG_INTERNAL
        else ""
    )

    console.print()

    console.print(
        Panel.fit(
            f"{mode}"
            "[bold cyan]ChatDSG Agent[/bold cyan]\n"
            "[dim]Terminal Mode[/dim]\n\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
            border_style="blue",
        )
    )

    console.print()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("ChatDSG agent (terminal mode)")

    parser.add_argument(
        "--scene-graph",
        nargs="?",
        const=None,
        default=None,
        help="DSG filepath to load",
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

    parser.add_argument(
        "--db_ip",
        type=str,
        help="Heracles database IP",
    )

    parser.add_argument(
        "--db_port",
        type=int,
        help="Heracles database port",
    )

    parser.add_argument(
        "--debug-internal",
        action="store_true",
        help=(
            "Show internal agent messages, "
            "tool outputs, and intermediate traces."
        ),
    )

    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save conversation history to disk",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Runtime flags
    # --------------------------------------------------------------------------

    global DEBUG_INTERNAL
    global SAVE_HISTORY

    DEBUG_INTERNAL = args.debug_internal
    SAVE_HISTORY = args.save_history

    # --------------------------------------------------------------------------
    # Env fallback
    # --------------------------------------------------------------------------

    if args.db_ip is None:
        args.db_ip = os.getenv("ADT4_HERACLES_IP")

    if args.db_port is None:
        args.db_port = os.getenv("ADT4_HERACLES_PORT")

    # --------------------------------------------------------------------------
    # Optional DSG loading
    # --------------------------------------------------------------------------

    if args.scene_graph:
        dsg_filepath = args.scene_graph

        console.print(
            Panel(
                f"[bold]Loading DSG[/bold]\n{dsg_filepath}",
                border_style="cyan",
            )
        )

        scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_filepath)

        summarize_dsg(scene_graph)

        neo4j_uri = f"neo4j://{args.db_ip}:{args.db_port}"

        neo4j_creds = (
            os.getenv("HERACLES_NEO4J_USERNAME"),
            os.getenv("HERACLES_NEO4J_PASSWORD"),
        )

        with console.status(
            "[bold blue]Loading scene graph into database..."
        ):
            load_dsg_to_db(
                args.object_labelspace,
                args.room_labelspace,
                neo4j_uri,
                neo4j_creds,
                scene_graph,
            )

        console.print(
            Panel(
                "[green]DSG successfully loaded.[/green]",
                border_style="green",
            )
        )

    # --------------------------------------------------------------------------
    # Load agent config
    # --------------------------------------------------------------------------

    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)

    # --------------------------------------------------------------------------
    # Suppress noisy startup prints unless debugging
    # --------------------------------------------------------------------------

    if DEBUG_INTERNAL:
        agent = LlmAgent(**yml)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = LlmAgent(**yml)

    # --------------------------------------------------------------------------
    # Initialize conversation
    # --------------------------------------------------------------------------

    messages = generate_initial_prompt(agent).to_anthropic_json(
        "Now you will interact with the user:"
    )

    print_welcome()

    # --------------------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------------------

    while True:
        try:
            user_input = console.input(
                "[bold green]You:[/bold green] "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        # ----------------------------------------------------------------------
        # Exit commands
        # ----------------------------------------------------------------------

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye.[/yellow]")
            break

        # Ignore empty inputs
        if not user_input:
            continue

        # ----------------------------------------------------------------------
        # Add user message
        # ----------------------------------------------------------------------

        messages += new_user_message(user_input)

        initial_length = len(messages)

        # ----------------------------------------------------------------------
        # Run agent
        # ----------------------------------------------------------------------

        cxt = AgentContext(agent)
        cxt.history = messages

        with console.status(
            "[bold blue]Assistant is thinking..."
        ):
            success, answer = cxt.run()

        # ----------------------------------------------------------------------
        # Error handling
        # ----------------------------------------------------------------------

        if not success:
            console.print(
                Panel(
                    "[red]Agent failed to produce a response.[/red]",
                    border_style="red",
                )
            )
            continue

        # ----------------------------------------------------------------------
        # Responses
        # ----------------------------------------------------------------------

        responses = cxt.get_agent_responses()

        new_responses = responses[initial_length:]

        # ----------------------------------------------------------------------
        # Debug: show all raw messages
        # ----------------------------------------------------------------------

        if DEBUG_INTERNAL:

            for idx, r in enumerate(new_responses):

                parsed = getattr(r, "parsed_response", None)

                if parsed is None:
                    continue

                parsed = str(parsed).strip()

                if not parsed:
                    continue

                print_internal_message(parsed)

        # ----------------------------------------------------------------------
        # Extract clean assistant response
        # ----------------------------------------------------------------------

        final_response = extract_final_response(new_responses)

        # ----------------------------------------------------------------------
        # Display final response
        # ----------------------------------------------------------------------

        if final_response:
            console.print(
                Panel(
                    Markdown(final_response),
                    title="Assistant",
                    border_style="blue",
                )
            )
        else:
            console.print(
                Panel(
                    "[yellow]No response generated.[/yellow]",
                    title="Assistant",
                    border_style="yellow",
                )
            )

        # ----------------------------------------------------------------------
        # Update history
        # ----------------------------------------------------------------------

        messages = cxt.history

        # ----------------------------------------------------------------------
        # Optional history saving
        # ----------------------------------------------------------------------

        if SAVE_HISTORY:
            save_history(messages)


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()