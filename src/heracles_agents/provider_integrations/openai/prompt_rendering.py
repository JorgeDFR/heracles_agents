import logging

logger = logging.getLogger(__name__)


def render_openai_example(example):
    parts = []

    if example.system:
        parts.append({"role": "developer", "content": example.system})

    parts.append({"role": "user", "content": example.user})
    parts.append({"role": "assistant", "content": example.assistant})
    return parts


def render_openai_prompt(prompt, novel_instruction=None):
    if prompt.novel_instruction is None and novel_instruction is None:
        raise ValueError(
            "novel_instruction must be set either at Prompt initialization or as an argument to `render_openai_prompt`"
        )
    rendered = [{"role": "developer", "content": prompt.system}]

    for field in (
        "scene_graph_description",
        "labelspace_description",
        "interface_description",
        "domain_description",
    ):
        value = getattr(prompt, field)
        if value:
            rendered.append({"role": "developer", "content": value})

    if prompt._api_prompt:
        logger.debug(f"Using API prompt: {prompt._api_prompt}")
        rendered.append({"role": "developer", "content": prompt._api_prompt})

    for value in (
        prompt.tool_description,
        prompt.in_context_examples_preamble,
    ):
        if value:
            rendered.append({"role": "developer", "content": value})

    if prompt.in_context_examples:
        logger.info(
            f"Adding {len(prompt.in_context_examples)} in-context examples to prompt"
        )
        for example in prompt.in_context_examples:
            rendered += render_openai_example(example)

    if prompt.novel_instruction_preamble:
        rendered.append({"role": "developer", "content": prompt.novel_instruction_preamble})

    if novel_instruction:
        if prompt.novel_instruction:
            logger.warning(
                f"Overriding default novel instruction `{prompt.novel_instruction}` with new instruction `{novel_instruction}`"
            )
        rendered.append({"role": "user", "content": novel_instruction})
    elif prompt.novel_instruction:
        rendered.append({"role": "user", "content": prompt.novel_instruction})

    if prompt.answer_semantic_guidance:
        rendered.append({"role": "developer", "content": prompt.answer_semantic_guidance})

    if prompt.answer_formatting_guidance:
        rendered.append(
            {"role": "developer", "content": prompt.answer_formatting_guidance}
        )

    return rendered
