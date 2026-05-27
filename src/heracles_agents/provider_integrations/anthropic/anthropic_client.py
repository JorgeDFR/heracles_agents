from typing import Literal

import anthropic
import logging

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


class AnthropicClientConfig(BaseSettings):
    client_type: Literal["anthropic"]
    auth_key: SecretStr = Field(alias="HERACLES_ANTHROPIC_API_KEY", exclude=True)
    _client: object = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        self._client = anthropic.Anthropic(api_key=self.auth_key.get_secret_value())

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise NotImplementedError(
                "Only `text` format is currently implemented for interfacing with Anthropic"
            )

        try:
            return self._client.messages.create(
                model=model_info.model,
                temperature=model_info.temperature,
                tools=tools,
                messages=messages,
                max_tokens=4096,
            )

        except anthropic.RateLimitError as ex:
            raise LlmRateLimitError(str(ex)) from ex

        except anthropic.APITimeoutError as ex:
            raise LlmTimeoutError(str(ex)) from ex

        except anthropic.APIConnectionError as ex:
            raise LlmConnectionError(str(ex)) from ex

        except anthropic.InternalServerError as ex:
            raise LlmServiceUnavailableError(str(ex)) from ex

        except anthropic.BadRequestError as ex:
            raise LlmBadRequestError(str(ex)) from ex

        except anthropic.AuthenticationError as ex:
            raise LlmAuthenticationError(str(ex)) from ex

        except Exception as ex:
            raise LlmUnknownError(str(ex)) from ex
