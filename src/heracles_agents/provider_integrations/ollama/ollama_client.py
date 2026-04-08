import os
from typing import Literal
from ollama import ChatResponse, chat, Client
from pydantic import PrivateAttr
from pydantic_settings import BaseSettings


class OllamaClientConfig(BaseSettings):
    client_type: Literal["ollama"]
    _chat_func: object = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        ollama_host = os.environ.get('OLLAMA_HOST', 'https://ollama.com')
        api_key = os.environ.get('OLLAMA_API_KEY')
        if ollama_host == 'https://ollama.com' and api_key:
            client = Client(
                host=ollama_host,
                headers={'Authorization': f'Bearer {api_key}'}
            )
            self._chat_func = client.chat
        elif self._chat_func is None:
            self._chat_func = chat

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise ValueError(
                f"response_format {response_format} not implemented for Ollama!"
            )

        options = {"temperature": model_info.temperature}
        if model_info.seed is not None:
            options["seed"] = model_info.seed

        response: ChatResponse = self._chat_func(
            model=model_info.model,
            messages=messages,
            tools=tools,
            think=False, # TODO: Validate this hack -> Some models take to much time in the thinking process
            options=options,
        )

        return response
