from heracles_agents.token_utils import estimate_text_tokens, register_text_token_counter


def count_anthropic_text_tokens(model_name, text):
    return estimate_text_tokens(text)


register_text_token_counter("anthropic", count_anthropic_text_tokens)
