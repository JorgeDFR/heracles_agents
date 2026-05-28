from heracles_agents.structured_tool_interface import StructuredToolDescription
from heracles_agents.tool_rendering import (
    register_tool_renderer,
    render_parameter_properties,
)


def render_openrouter_tool(tool):
    if isinstance(tool, StructuredToolDescription):
        raise NotImplementedError(
            "Structured tool calling not supported for OpenRouter tools"
        )

    parameter_descriptions = {
        "type": "object",
        "properties": render_parameter_properties(tool),
        "required": [p.name for p in tool.parameters if p.required],
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameter_descriptions,
        },
    }


register_tool_renderer("openrouter", render_openrouter_tool)
