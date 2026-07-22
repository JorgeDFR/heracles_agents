from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.tool_calling.registry import ToolRegistry, register_tool
from pypddl.pddl_goal_parser import get_pddl_goal_lark_grammar

import logging
logger = logging.getLogger(__name__)


pddl_tool = StructuredToolDescription(
    name="pddl_answer_tool",
    description="Use this tool to submit your final PDDL goal.",
    grammar=get_pddl_goal_lark_grammar(),
)

register_tool(pddl_tool)
logger.debug(f"Registered tools: {ToolRegistry.registered_tool_summary()}")
