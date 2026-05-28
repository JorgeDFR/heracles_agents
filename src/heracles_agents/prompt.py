import logging
import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, PrivateAttr, field_validator

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
    in_context_examples: Optional[List[InContextExample]] = None
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
                logger.info(f"Loading {path}")
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
    base_prompt: Prompt
    output_type: Optional[str] = None
    answer_type_hint: bool = False
    sldp_answer_type_hint: bool = False

    @property
    def include_answer_type_hint(self):
        return self.answer_type_hint or self.sldp_answer_type_hint

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


def get_sldp_format_description():
    return """
Please format your response according to the SLDP Equality Language:

## SLDP Equality Language

To evaluate if an answer is correct, we need to define a sense of equality.
This is rather tricky, because there are different senses in which things can be equal.

We need to handle Lists, Sets, Dictionaries, and Points.
Lists are equal if each element is equal.
Sets A and B are equal if A ⊆ B and B ⊆ A.
Dictionaries are equal if the sets of their keys are equal and the value for each key matches between dictionaries.
Two points are equal if they are within some tolerance.
Of course primitive numbers and strings can also be compared for equality.
We support arbitrary compositions of these containers.

We expect nodes in the graph to be represented without any parentheses.
For example O(1) should be represented as O1.
We also expect no additional information than what is explicitly asked for in the question.
E.g., if the question asks for a list of node IDs, the answer should be a list of node IDs and not a list of nodes with their properties or if the question asks for locations a list of points should be provided and not a list of nodes with their locations.

### Syntax

A primitive string is a sequence of alphanumeric characters (with no quotation).

A primitive number is a floating point representation of a number.

A `list` is written as `[element1, element2, ... elementN]`

A `set` is written as `<element1, element2, ... elementN>`

A `dict` is written as `{k1: v1, k2: v2}`

A `point` is written as `POINT(x y z)` (note the lack of comma)
"""


def get_sldp_answer_tag_text():
    return """
### Denoting Final Answer:

Format your final answer (*not* any intermediate tool calls) as an SLDP expression wrapped between the <answer> and </answer> tags (XML-style format).
Example: <answer> <1,2,3> </answer>
Only a single pair of answer tags should appear in your solution.
"""
