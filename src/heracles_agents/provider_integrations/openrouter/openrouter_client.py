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

        response = self._client.chat.send(**payload)

        print(response)
        print()
        print(response.choices[0])
        print()
        print(response.choices[0].message)
        print()
        print(response.choices[0].message.content)
        print()
        print(response.choices[0].message.tool_calls)
        print()

        # TODO: find the type of response and how to get tool call component and type

        return response