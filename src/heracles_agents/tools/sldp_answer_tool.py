from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.tool_calling.registry import ToolRegistry, register_tool
from sldp.lark_parser import get_sldp_lark_grammar

import logging
logger = logging.getLogger(__name__)

sldp_tool = StructuredToolDescription(
    name="sldp_answer_tool",
    description="Use this tool to submit your final SLDP-formatted answer.",
    grammar=get_sldp_lark_grammar(),
)

register_tool(sldp_tool)
logger.debug(f"Registered tools: {ToolRegistry.registered_tool_summary()}")
