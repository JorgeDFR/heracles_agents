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
            logger.error("All retries failed.")
            TODO

        return response




from typing import Literal

import logging
from openrouter import OpenRouter

from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings

from heracles_agents.exceptions import (
    LlmRateLimitError,
    LlmTimeoutError,
    LlmServiceUnavailableError,
    LlmAuthenticationError,
    LlmConnectionError,
    LlmBadRequestError,
    LlmUnknownError,
)

logger = logging.getLogger(__name__)


class OpenRouterClientConfig(BaseSettings):
    client_type: Literal["openrouter"]
    auth_key: SecretStr = Field(alias="HERACLES_OPENROUTER_API_KEY", exclude=True)
    _client: OpenRouter = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenRouter(api_key=self.auth_key.get_secret_value())


    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise ValueError(
                f"response_format {response_format} "
                f"not implemented for OpenRouter!"
            )

        payload = self._build_payload(
            model_info,
            tools,
            messages,
        )

        try:
            return self._client.chat.send(**payload)
        except Exception as ex:
            self._raise_normalized_error(ex)


    def _build_payload(self, model_info, tools, messages):
        payload = {
            "model": model_info.model,
            "messages": messages,
            "tools": tools,
            "temperature": model_info.temperature,
        }

        if getattr(model_info, "seed", None) is not None:
            payload["seed"] = model_info.seed

        if getattr(model_info, "reasoning", None) is not None:
            payload["reasoning"] = {
                "effort": model_info.reasoning
            }

        return payload


    def _raise_normalized_error(self, ex: Exception):
        msg = str(ex).lower()

        logger.error(f"OpenRouter request failed: {ex}")

        # ----------------------------------------------------------
        # Rate limit
        # ----------------------------------------------------------
        if "429" in msg or "rate limit" in msg:
            raise LlmRateLimitError(str(ex)) from ex

        # ----------------------------------------------------------
        # Timeout
        # ----------------------------------------------------------
        if "timeout" in msg:
            raise LlmTimeoutError(str(ex)) from ex

        # ----------------------------------------------------------
        # Auth
        # ----------------------------------------------------------
        if "auth" in msg or "unauthorized" in msg:
            raise LlmAuthenticationError(str(ex)) from ex

        # ----------------------------------------------------------
        # Bad request
        # ----------------------------------------------------------
        if "invalid" in msg or "bad request" in msg:
            raise LlmBadRequestError(str(ex)) from ex

        # ----------------------------------------------------------
        # Connection issues
        # ----------------------------------------------------------
        if "connection" in msg or "network" in msg:
            raise LlmConnectionError(str(ex)) from ex

        # ----------------------------------------------------------
        # Fallback
        # ----------------------------------------------------------
        raise LlmUnknownError(str(ex)) from ex