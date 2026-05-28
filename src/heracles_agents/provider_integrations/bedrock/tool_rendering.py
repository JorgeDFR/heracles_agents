from heracles_agents.structured_tool_interface import StructuredToolDescription
from heracles_agents.tool_rendering import (
    register_tool_renderer,
    render_parameter_properties,
)


def render_bedrock_tool(tool):
    if isinstance(tool, StructuredToolDescription):
        raise NotImplementedError("Structured tool calling not supported for Bedrock tools")

    input_schema = {
        "json": {
            "type": "object",
            "properties": render_parameter_properties(tool),
            "required": [p.name for p in tool.parameters if p.required],
        }
    }
    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": input_schema,
        }
    }


register_tool_renderer("bedrock", render_bedrock_tool)
