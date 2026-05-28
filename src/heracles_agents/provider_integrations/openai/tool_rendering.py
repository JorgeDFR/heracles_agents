from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.tool_calling.rendering import (
    register_tool_renderer,
    render_parameter_properties,
)


def render_openai_tool(tool):
    if isinstance(tool, StructuredToolDescription):
        return {
            "type": "custom",
            "name": tool.name,
            "description": tool.description,
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": tool.grammar,
            },
        }

    parameter_descriptions = {
        "type": "object",
        "properties": render_parameter_properties(tool),
        "required": [p.name for p in tool.parameters if p.required],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameter_descriptions,
    }


register_tool_renderer("openai", render_openai_tool)
