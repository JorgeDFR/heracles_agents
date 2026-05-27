from typing import Literal

import openai
import logging

from pydantic import BaseModel, Field, PrivateAttr, SecretStr
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


class OpenaiClientConfig(BaseSettings):
    client_type: Literal["openai"]
    timeout: int
    auth_key: SecretStr = Field(alias="HERACLES_OPENAI_API_KEY", exclude=True)
    _client: object = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        self._client = openai.OpenAI(
            api_key=self.auth_key.get_secret_value(), timeout=self.timeout
        )


    def call(self, model_info, tools, response_format, messages):
        payload = self._build_payload(
            model_info=model_info,
            tools=tools,
            response_format=response_format,
            messages=messages,
        )

        try:
            return self._client.responses.create(**payload)

        # ------------------------------------------------------------------
        # Retryable
        # ------------------------------------------------------------------
        except openai.RateLimitError as ex:
            raise LlmRateLimitError(str(ex)) from ex

        except openai.APITimeoutError as ex:
            raise LlmTimeoutError(str(ex)) from ex

        except openai.APIConnectionError as ex:
            raise LlmConnectionError(str(ex)) from ex

        except openai.InternalServerError as ex:
            raise LlmServiceUnavailableError(str(ex)) from ex

        # ------------------------------------------------------------------
        # Fatal
        # ------------------------------------------------------------------
        except openai.AuthenticationError as ex:
            raise LlmAuthenticationError(str(ex)) from ex

        except openai.BadRequestError as ex:
            raise LlmBadRequestError(str(ex)) from ex

        # ------------------------------------------------------------------
        # Unknown
        # ------------------------------------------------------------------
        except Exception as ex:
            logger.exception("Unexpected OpenAI provider error")
            raise LlmUnknownError(str(ex)) from ex


    def _build_payload(self, model_info,  tools, response_format, messages):
        payload = {
            "model": model_info.model,
            "temperature": model_info.temperature,
            "text": self._build_response_format(
                response_format
            ),
            "tools": tools,
            "input": messages,
            "parallel_tool_calls": False,
        }

        # Optional parameters
        if getattr(model_info, "seed", None) is not None:
            payload["seed"] = model_info.seed

        # GPT-5 reasoning models
        if "gpt-5" in model_info.model:

            payload["reasoning"] = {
                "effort": getattr(
                    model_info,
                    "reasoning",
                    "low",
                )
            }

        return payload


    def _build_response_format(self, response_format,):
        match response_format:
            case "text":
                return {"format": {"type": "text"}}
            case "json":
                return {"format": {"type": "json_object"}}
            case BaseModel():
                return response_format
            case _:
                raise ValueError(
                    "Unknown response_format "
                    f"requested for LLM: "
                    f"{response_format}"
                )