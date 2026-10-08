import copy
from functools import partial
from typing import Any, Generic, Optional, TypeVar

from plum import parametric
from pydantic import BaseModel, Field, field_serializer, field_validator

from heracles_agents.inference_parameters import ReasoningSettings
from heracles_agents.provider_integrations.model_client_interfaces import get_client_union_type
from heracles_agents.prompt import PromptSettings
from heracles_agents.pydantic_discriminated_dispatch import (
    discriminated_union_dispatch,
)
from heracles_agents.tool_calling.structured_tool_description import StructuredToolDescription
from heracles_agents.tool_calling.tool_description import ToolDescription
from heracles_agents.tool_calling.registry import ToolRegistry


class ModelInfo(BaseModel):
    """Settings that affect fundamental model performance.

    e.g., model size, temperature, seed.
    Tool calling details are handled elsewhere
    """

    model: str
    temperature: Optional[float] = 1.0
    seed: Optional[int] = None
    reasoning: Optional[ReasoningSettings] = None
    openrouter_provider: Optional[str] = None
    parameter_capabilities: dict[str, Any] = Field(default_factory=dict)
    response_format: str = "text"

    @field_validator("openrouter_provider")
    @classmethod
    def validate_openrouter_provider(cls, value):
        if value is not None:
            if not value.strip():
                raise ValueError(
                    "openrouter_provider must be a non-empty string or null"
                )
            return value.strip()
        return None

    @field_validator("reasoning", mode="before")
    @classmethod
    def normalize_legacy_reasoning(cls, value):
        if isinstance(value, str):
            return {"mode": "enabled", "effort": value}
        if isinstance(value, bool):
            return {"mode": "enabled" if value else "disabled"}
        return value


def apply_bound_args(tool_name, bound_args):
    args_to_bind = {}
    for arg_name, fields in bound_args.items():
        arg_type = ToolRegistry.get_arg_type(tool_name, arg_name)
        if arg_type is str or arg_type is int or arg_type is float:
            arg_instance = arg_type(fields)
        else:
            arg_instance = arg_type(**fields)
        args_to_bind[arg_name] = arg_instance
    function = partial(ToolRegistry.tools[tool_name].function, **args_to_bind)
    return function, args_to_bind


class AgentInfo(BaseModel):
    """Configuration for "agentic" behaviors, e.g., tool calling"""

    prompt_settings: PromptSettings
    tools: dict[str, ToolDescription | StructuredToolDescription]
    tool_interface: str
    max_iterations: int

    @field_validator("tools", mode="before")
    @classmethod
    def lookup_tools(cls, tools):
        tool_descriptions = {}
        for t in tools:
            tool_name = t["name"]
            if tool_name not in ToolRegistry.tools:
                raise ValueError(
                    f"Unknown tool {tool_name}. Known tools: {list(ToolRegistry.tools.keys())}"
                )
            if "bound_args" in t:
                function, validated_args = apply_bound_args(tool_name, t["bound_args"])
                resolved_tool = copy.deepcopy(ToolRegistry.tools[tool_name])
                resolved_tool.function = function
                resolved_tool._bound_args = validated_args
            else:
                resolved_tool = ToolRegistry.tools[tool_name]
            tool_descriptions[tool_name] = resolved_tool

        return tool_descriptions

    @field_serializer("tools")
    def serialize_tools(
        self, tools: dict[str, ToolDescription | StructuredToolDescription]
    ):
        def dump(x):
            if isinstance(x, BaseModel):
                return x.model_dump()
            return x

        serialized_tools = []
        for tool in tools.values():
            d = {"name": tool.name}
            if tool._bound_args is not None:
                d["bound_args"] = {
                    argname: dump(val) for argname, val in tool._bound_args.items()
                }
            serialized_tools.append(d)
        return serialized_tools


model_interface_config_type = get_client_union_type()
T = TypeVar("T")


@parametric
@discriminated_union_dispatch("client")
class LlmAgent(BaseModel, Generic[T]):
    agent_info: AgentInfo
    model_info: ModelInfo
    client: model_interface_config_type = Field(discriminator="client_type")
