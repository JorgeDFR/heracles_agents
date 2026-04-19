from typing import Literal

from openrouter import OpenRouter
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings


class OpenRouterClientConfig(BaseSettings):
    client_type: Literal["openrouter"]
    auth_key: SecretStr = Field(alias="HERACLES_OPENROUTER_API_KEY", exclude=True)
    _client: OpenRouter = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenRouter(
            api_key=self.auth_key.get_secret_value(),
        )

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise ValueError(
                f"response_format {response_format} not implemented for OpenRouter!"
            )

        payload = {
            "model": model_info.model,
            "messages": messages,
            "tools": tools,
            "temperature": model_info.temperature,
        }

        if model_info.seed is not None:
            payload["seed"] = model_info.seed

        try:
            response = self._client.chat.send(**payload)
            print("Test 123: ", response)
        except Exception as e:
            print("Test 123: ", e)

        # TODO: deal with erros such as:
        # - POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"

        return response