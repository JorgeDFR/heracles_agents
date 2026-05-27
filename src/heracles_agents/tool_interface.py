import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, PrivateAttr, model_validator

from heracles_agents.tool_registry import ToolRegistry


def type_to_string(typ):
    match typ():
        case str():
            return "string"
        case float():
            return "number"
        case int():
            return "integer"
        case dict():
            return "object"
        case set():
            return "array"
        case list():
            return "array"


@dataclass
class FunctionParameter:
    """Description of a single parameter for a tool/function call"""

    name: str
    param_type: type
    param_description: str
    required: bool = True
    enum_values: Optional[Any] = None


class ToolDescription(BaseModel):
    """Description of a tool / function"""

    name: str
    description: str
    parameters: list[FunctionParameter]
    function: Callable
    _bound_args: PrivateAttr() = None  # dict[str, object]

    def get_tool_function(self):
        try:
            fn = ToolRegistry.tools[self.name]
        except IndexError as ex:
            print(ex)
            print(
                f"Tool {self.name} not registered in ToolRegistry! Registered tools are {ToolRegistry.registered_tool_summary()}"
            )
        return fn

    @model_validator(mode="after")
    def verify_param_names(self):
        function_required_params = set()
        function_optional_params = set()
        for k, v in inspect.signature(self.function).parameters.items():
            if v.default == inspect._empty:
                function_required_params.add(k)
            else:
                function_optional_params.add(k)

        given_param_names = set(p.name for p in self.parameters)

        for gp in given_param_names:
            if gp not in function_optional_params.union(function_required_params):
                raise ValueError(
                    f"Declared parameter {gp} is not an optional or required keyword in defined function"
                )

        if not function_required_params <= given_param_names:
            missing_params = function_required_params - given_param_names
            raise ValueError(
                f"Function requires parameters that are not declared by tool: {missing_params}"
            )

        return self
