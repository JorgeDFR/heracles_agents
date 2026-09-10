import logging
import os
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

logger = logging.getLogger(__name__)


class InContextExample(BaseModel):
    user: str
    assistant: str
    system: Optional[str] = None


class Prompt(BaseModel):
    system: str
    interface_description: Optional[str] = None
    scene_graph_description: Optional[str] = None
    labelspace_description: Optional[str] = None
    domain_description: Optional[str] = None
    tool_description: Optional[str] = None
    in_context_examples_preamble: Optional[str] = None
    # A text block is rendered as one user message. The structured list keeps
    # the legacy alternating user/assistant demonstration format.
    in_context_examples: Optional[Union[str, List[InContextExample]]] = None
    novel_instruction_preamble: Optional[str] = None
    novel_instruction: Optional[str] = None
    novel_instruction_template: Optional[str] = None
    answer_semantic_guidance: Optional[str] = None
    answer_formatting_guidance: Optional[str] = None

    _api_prompt: PrivateAttr() = None

    def set_api_prompt(self, api_prompt):
        self._api_prompt = api_prompt

    @field_validator(
        "scene_graph_description",
        "interface_description",
        "domain_description",
        "labelspace_description",
        "in_context_examples",
        mode="before",
    )
    @classmethod
    def load_description_from_yaml(cls, value: Optional[str], info):
        """
        If the field points to a YAML file, load it and extract the content
        under the corresponding field name. Otherwise return the value as-is.
        """
        if isinstance(value, str):
            path = os.path.expandvars(value)
            if path.endswith((".yaml", ".yml")):
                if not os.path.isfile(path):
                    raise ValueError(f"Description YAML path does not exist: {path}")
                logger.debug(f"Loading {path}")
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                loaded_data = data.get(info.field_name, None)
                if loaded_data is None:
                    logger.error(
                        f"Failed to load {info.field_name}. Only found {list(data.keys())}"
                    )
                return loaded_data
        return value

    def __repr__(self):
        return repr(self.model_dump())


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_prompt: Prompt
    output_type: Optional[str] = None
    answer_type_hint: bool = False

    @property
    def include_answer_type_hint(self):
        return self.answer_type_hint

    @field_validator("base_prompt", mode="before")
    @classmethod
    def load_prompt(cls, prompt_path):
        match prompt_path:
            case str():
                prompt_path = os.path.expandvars(prompt_path)
                if not os.path.exists(prompt_path):
                    raise ValueError(f"Prompt path does not exist: {prompt_path}")
                with open(prompt_path, "r") as fo:
                    prompt_yaml = yaml.safe_load(fo)
                return Prompt(**prompt_yaml)
            case dict():
                return prompt_path
            case Prompt():
                return prompt_path
            case _:
                raise ValueError(
                    f"PromptSettings cannot initialize base_prompt from type {type(prompt_path)}"
                )
