import logging

logger = logging.getLogger(__name__)


def _text_message(role, text):
    return {"role": role, "content": [{"text": text}]}


def render_bedrock_example(example):
    if example.system:
        logger.error("Bedrock does not support system tags for in-context examples!")

    return [
        _text_message("user", example.user),
        _text_message("assistant", example.assistant),
    ]


def render_bedrock_prompt(prompt, novel_instruction=None):
    if prompt.novel_instruction is None and novel_instruction is None:
        raise ValueError(
            "novel_instruction must be set either at Prompt initialization or as an argument to `render_bedrock_prompt`"
        )
    rendered = [_text_message("user", prompt.system)]

    for field in (
        "scene_graph_description",
        "labelspace_description",
        "interface_description",
        "domain_description",
    ):
        value = getattr(prompt, field)
        if value:
            rendered.append(_text_message("user", value))

    if prompt._api_prompt:
        logger.debug(f"Using API prompt: {prompt._api_prompt}")
        rendered.append(_text_message("user", prompt._api_prompt))

    for value in (
        prompt.tool_description,
        prompt.in_context_examples_preamble,
    ):
        if value:
            rendered.append(_text_message("user", value))

    if prompt.in_context_examples:
        for example in prompt.in_context_examples:
            rendered += render_bedrock_example(example)

    if prompt.novel_instruction_preamble:
        rendered.append(_text_message("user", prompt.novel_instruction_preamble))

    if novel_instruction:
        if prompt.novel_instruction:
            logger.warning(
                f"Overriding default novel instruction `{prompt.novel_instruction}` with new instruction `{novel_instruction}`"
            )
        rendered.append(_text_message("user", novel_instruction))
    elif prompt.novel_instruction:
        rendered.append(_text_message("user", prompt.novel_instruction))

    if prompt.answer_semantic_guidance:
        rendered.append(_text_message("user", prompt.answer_semantic_guidance))

    if prompt.answer_formatting_guidance:
        rendered.append(_text_message("user", prompt.answer_formatting_guidance))

    return rendered
