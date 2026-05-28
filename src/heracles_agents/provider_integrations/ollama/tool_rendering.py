from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.tool_calling.rendering import (
    register_tool_renderer,
    render_parameter_properties,
)


def render_ollama_tool(tool):
    if isinstance(tool, StructuredToolDescription):
        raise NotImplementedError("Structured tool calling not supported for Ollama tools")

    parameter_descriptions = {
        "type": "object",
        "properties": render_parameter_properties(tool),
        "required": [p.name for p in tool.parameters if p.required],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameter_descriptions,
        },
    }


register_tool_renderer("ollama", render_ollama_tool)
