from heracles_agents.structured_tool_interface import StructuredToolDescription
from heracles_agents.tool_interface import FunctionParameter, ToolDescription, type_to_string


def render_parameter(parameter: FunctionParameter):
    rendered = {
        parameter.name: {
            "type": type_to_string(parameter.param_type),
            "description": parameter.param_description,
        }
    }
    if parameter.enum_values is not None:
        rendered[parameter.name]["enum"] = parameter.enum_values
    return rendered


def render_custom_parameter(parameter: FunctionParameter):
    rendered = [
        f"Param name: {parameter.name}",
        f"Param description: {parameter.param_description}",
        f"type: {type_to_string(parameter.param_type)}",
    ]
    if parameter.enum_values is not None:
        rendered.append(f"Allowed values: {str(parameter.enum_values)}")
    return rendered


def render_tool_for_interface(tool, tool_interface):
    match tool_interface:
        case "openai":
            return render_openai_tool(tool)
        case "anthropic":
            return render_anthropic_tool(tool)
        case "ollama":
            return render_ollama_tool(tool)
        case "bedrock":
            return render_bedrock_tool(tool)
        case "openrouter":
            return render_openrouter_tool(tool)
        case _:
            raise NotImplementedError(f"Unknown tool interface: {tool_interface}")


def render_parameter_properties(tool: ToolDescription):
    parameter_properties = {}
    for parameter in tool.parameters:
        parameter_properties |= render_parameter(parameter)
    return parameter_properties


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


def render_anthropic_tool(tool):
    if isinstance(tool, StructuredToolDescription):
        raise NotImplementedError(
            "Structured tool calling not supported for Anthropic tools"
        )

    parameter_descriptions = {
        "type": "object",
        "properties": render_parameter_properties(tool),
        "required": [p.name for p in tool.parameters if p.required],
    }
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": parameter_descriptions,
    }


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


def render_custom_tool(tool):
    if isinstance(tool, StructuredToolDescription):
        raise NotImplementedError("Structured tool calling not supported for custom tools")

    parameter_descriptions = ""
    for parameter in tool.parameters:
        rendered = "\n".join(["  " + s for s in render_custom_parameter(parameter)])
        rendered = "-" + rendered[1:]
        parameter_descriptions += rendered + "\n"
    return f"""
Function name: {tool.name}
Function Description: {tool.description}
Parameters:
{parameter_descriptions}
"""
