import logging

import tiktoken

from heracles_agents.token_utils import estimate_text_tokens, register_text_token_counter

logger = logging.getLogger(__name__)


def normalize_openai_model_name(model_name):
    if model_name and "gpt-5" in model_name:
        return "gpt-5-latest"
    return model_name


def count_openai_text_tokens(model_name, text):
    model_name = normalize_openai_model_name(model_name)
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(
            "No tiktoken encoder for OpenAI model %r; using local token estimate",
            model_name,
        )
        return estimate_text_tokens(text)
    return len(enc.encode(str(text)))


register_text_token_counter("openai", count_openai_text_tokens)
