import logging

logger = logging.getLogger(__name__)


def render_huggingface_example(example):
    parts = []

    if example.system:
        parts.append({"role": "system", "content": example.system})

    parts.append({"role": "user", "content": example.user})
    parts.append({"role": "assistant", "content": example.assistant})
    return parts


def render_huggingface_prompt(prompt, novel_instruction=None):
    if prompt.novel_instruction is None and novel_instruction is None:
        raise ValueError(
            "novel_instruction must be set either at Prompt initialization or as "
            "an argument to `render_huggingface_prompt`"
        )
    rendered = [{"role": "system", "content": prompt.system}]

    developer_parts = []
    for field in (
        "scene_graph_description",
        "labelspace_description",
        "interface_description",
        "domain_description",
    ):
        value = getattr(prompt, field)
        if value:
            developer_parts.append(value)

    if prompt._api_prompt:
        logger.debug(f"Using API prompt: {prompt._api_prompt}")
        developer_parts.append(prompt._api_prompt)

    for value in (
        prompt.tool_description,
        prompt.in_context_examples_preamble,
    ):
        if value:
            developer_parts.append(value)

    if developer_parts:
        rendered.append({"role": "system", "content": "\n\n".join(developer_parts)})

    if prompt.in_context_examples:
        if isinstance(prompt.in_context_examples, str):
            logger.info("Adding in-context examples block to prompt")
            rendered.append({"role": "user", "content": prompt.in_context_examples})
        else:
            logger.info(
                f"Adding {len(prompt.in_context_examples)} in-context examples to prompt"
            )
            for example in prompt.in_context_examples:
                rendered += render_huggingface_example(example)

    user_parts = []
    if prompt.novel_instruction_preamble:
        user_parts.append(prompt.novel_instruction_preamble)

    if novel_instruction:
        if prompt.novel_instruction:
            logger.warning(
                f"Overriding default novel instruction `{prompt.novel_instruction}` "
                f"with new instruction `{novel_instruction}`"
            )
        user_parts.append(novel_instruction)
    elif prompt.novel_instruction:
        user_parts.append(prompt.novel_instruction)

    if prompt.answer_semantic_guidance:
        user_parts.append(prompt.answer_semantic_guidance)

    if prompt.answer_formatting_guidance:
        user_parts.append(prompt.answer_formatting_guidance)

    if user_parts:
        rendered.append({"role": "user", "content": "\n\n".join(user_parts)})

    return rendered
