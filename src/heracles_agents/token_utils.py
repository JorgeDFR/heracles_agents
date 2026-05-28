import logging
import math
import re

logger = logging.getLogger(__name__)

_TEXT_TOKEN_COUNTERS = {}


def register_text_token_counter(provider, counter):
    _TEXT_TOKEN_COUNTERS[provider] = counter


def estimate_text_tokens(text):
    if text is None:
        return 0
    text = str(text)
    if not text:
        return 0

    lexical_units = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    character_estimate = math.ceil(len(text) / 4)
    return max(len(lexical_units), character_estimate)


def get_token_provider(agent):
    client = getattr(agent, "client", None)
    provider = getattr(client, "client_type", None)
    if provider is not None:
        return provider
    agent_info = getattr(agent, "agent_info", None)
    return getattr(agent_info, "tool_interface", None)


def count_text_tokens(agent, text):
    provider = get_token_provider(agent)
    model = getattr(getattr(agent, "model_info", None), "model", None)
    counter = _TEXT_TOKEN_COUNTERS.get(provider)
    if counter is None:
        logger.debug(
            "No tokenizer registered for provider %r and model %r; using local estimate",
            provider,
            model,
        )
        return estimate_text_tokens(text)
    return counter(model, text)
