import time
import logging
from typing import Literal

from openrouter import OpenRouter
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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

        if model_info.reasoning is not None:
            payload["reasoning"] = {"effort": model_info.reasoning}

        max_retries = 3
        backoff_factor = 2
        response = None
        retries = 0
        while retries <= max_retries:
            try:
                response = self._client.chat.send(**payload)
                break
            except Exception as e:
                retries += 1
                if "429" in str(e):
                    wait_time = backoff_factor ** retries
                    logger.warning(
                        f"Rate limit hit (HTTP 429). Retrying in {wait_time} seconds... [Attempt {retries}/{max_retries}]"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed: {e}")
                    break

        if response is None:
            response = {"error": "Request failed after retries."}
            logger.error("All retries failed. Returning error response.")

        return response