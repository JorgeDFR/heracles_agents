from heracles_agents.provider_integrations.openai.token_counting import (
    count_openai_text_tokens,
)
from heracles_agents.token_utils import estimate_text_tokens, register_text_token_counter


def count_openrouter_text_tokens(model_name, text):
    if model_name and model_name.startswith("openai/"):
        return count_openai_text_tokens(model_name.removeprefix("openai/"), text)
    return estimate_text_tokens(text)


register_text_token_counter("openrouter", count_openrouter_text_tokens)
