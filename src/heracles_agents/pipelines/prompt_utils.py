from sldp.sldp_lang import get_sldp_type


def get_sldp_format_description():
    return """
Please format your response according to the SLDP Language:

## SLDP Language

A SLDP answer represents the answer to a direct question.
It can be a primitive string, primitive number, list, set, dictionarie, or point.
It can also be an arbitrary composition of these containers.

Use object and place symbols exactly as they appear in the scene graph. Node symbols should not contain parentheses or quotes. For example, O(1) should be written as O1.

Do not include additional information than what is explicitly asked for in the question.
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
Only a single pair of answer tags should appear.
"""


def include_answer_type_hint(prompt_settings):
    return getattr(
        prompt_settings,
        "include_answer_type_hint",
        getattr(prompt_settings, "answer_type_hint", False),
    )


def get_pddl_format_description():
    return """
Please format your response according to the PDDL Goal Language:

## PDDL Goal Language

A PDDL goal describes the desired final state for a planner. It must be a single valid goal expression using predicates from the provided PDDL domain.

Use object and place symbols exactly as they appear in the scene graph. Node symbols should not contain parentheses or quotes. For example, O(1) should be written as O1.

Do not include explanatory text inside the final answer. Do not invent predicates outside the provided PDDL domain.

### Syntax

An atomic predicate is written as `(predicate arg1 arg2 ... argN)`.
A conjunction is written as `(and goal1 goal2 ... goalN)`.
A disjunction is written as `(or goal1 goal2 ... goalN)`.
A negated atomic predicate is written as `(not (predicate arg1 arg2 ... argN))`.

Examples:
`(visited-object O1)`
`(and (visited-place R0) (visited-place R1))`
`(or (visited-object O2) (visited-object O5))`
"""


def get_pddl_answer_tag_text():
    return """
### Denoting Final Answer:

Format your final answer (*not* any intermediate tool calls) as a PDDL goal wrapped between the <answer> and </answer> tags (XML-style format).
Example: <answer> (visited-place P100) </answer>
Only a single pair of answer tags should appear.
"""


def get_answer_formatting_guidance_helper(prompt_settings, question):
    match prompt_settings.output_type:
        case "SLDP":
            format_instruction = get_sldp_format_description()
            format_instruction += get_sldp_answer_tag_text()
            if include_answer_type_hint(prompt_settings):
                sldp_type = get_sldp_type(question.solution)
                sldp_type_lower = sldp_type.lower()
                if sldp_type_lower == "string":
                    sldp_type = "primitive string"
                elif sldp_type_lower == "number":
                    sldp_type = "primitive number"
                format_instruction += f"\n Your answer should be an SLDP {sldp_type}"
            return format_instruction
        case "SLDP_TOOL":
            format_instruction = get_sldp_format_description()
            if include_answer_type_hint(prompt_settings):
                sldp_type = get_sldp_type(question.solution)
                format_instruction += f"\n Your answer should be an SLDP {sldp_type}"

            format_instruction += (
                "\n Call the tool sldp_answer_tool to submit your final answer."
            )
            return format_instruction
        case "PDDL":
            format_instruction = get_pddl_format_description()
            format_instruction += get_pddl_answer_tag_text()
            return format_instruction
        case "PDDL_TOOL":
            format_instruction = get_pddl_format_description()
            format_instruction += (
                "\n Call the tool pddl_answer_tool to submit your final answer."
            )
            return format_instruction
        case None:
            # The "default". Presumably the description of the output format is
            # in the base prompt.
            return None
        case _:
            raise ValueError(f"Unknown output type: {prompt_settings.output_type}")


def get_answer_formatting_guidance(agent_config, question):
    return get_answer_formatting_guidance_helper(
        agent_config.agent_info.prompt_settings, question
    )
