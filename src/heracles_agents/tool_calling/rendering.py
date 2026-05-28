from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription, type_to_string

_TOOL_RENDERERS = {}


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
    try:
        renderer = _TOOL_RENDERERS[tool_interface]
    except KeyError as ex:
        raise NotImplementedError(f"Unknown tool interface: {tool_interface}") from ex
    return renderer(tool)


def register_tool_renderer(tool_interface, renderer):
    _TOOL_RENDERERS[tool_interface] = renderer


def has_tool_renderer(tool_interface):
    return tool_interface in _TOOL_RENDERERS


def render_parameter_properties(tool: ToolDescription):
    parameter_properties = {}
    for parameter in tool.parameters:
        parameter_properties |= render_parameter(parameter)
    return parameter_properties


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
