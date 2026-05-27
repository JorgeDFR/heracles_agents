from typing import Literal

import boto3
import logging

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings

from heracles_agents.exceptions import (
    LlmRateLimitError,
    LlmTimeoutError,
    LlmServiceUnavailableError,
    LlmConnectionError,
    LlmBadRequestError,
    LlmAuthenticationError,
    LlmUnknownError,
)

logger = logging.getLogger(__name__)


model_name_to_bedrock_model_id = {
    "bedrock_claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock_claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "bedrock_claude-opus-4-1": "us.anthropic.claude-opus-4-1-20250805-v1:0",
}


class BedrockClientConfig(BaseSettings):
    client_type: Literal["bedrock"]
    _client: object = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        self._client = boto3.client("bedrock-runtime", region_name="us-east-1")


    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise NotImplementedError(
                "Only `text` format is currently implemented for interfacing with Bedrock"
            )

        model_id = model_name_to_bedrock_model_id[model_info.model]

        payload = self._build_payload(
            model_id=model_id,
            model_info=model_info,
            tools=tools,
            messages=messages,
        )

        try:
            return self._client.converse(**payload)

        # --------------------------------------------------------------
        # Retryable errors
        # --------------------------------------------------------------
        except self._client.exceptions.ThrottlingException as ex:
            raise LlmRateLimitError(str(ex)) from ex

        except self._client.exceptions.ModelTimeoutException as ex:
            raise LlmTimeoutError(str(ex)) from ex

        except self._client.exceptions.ServiceUnavailableException as ex:
            raise LlmServiceUnavailableError(str(ex)) from ex

        except self._client.exceptions.ValidationException as ex:
            raise LlmBadRequestError(str(ex)) from ex

        except self._client.exceptions.AccessDeniedException as ex:
            raise LlmAuthenticationError(str(ex)) from ex

        # --------------------------------------------------------------
        # Network / transport (boto3 side)
        # --------------------------------------------------------------
        except boto3.exceptions.Boto3Error as ex:
            raise LlmConnectionError(str(ex)) from ex

        # --------------------------------------------------------------
        # Unknown
        # --------------------------------------------------------------
        except Exception as ex:
            logger.exception("Unexpected Bedrock provider error")
            raise LlmUnknownError(str(ex)) from ex


    def _build_payload(self, model_id, model_info, tools, messages):
        req = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": model_info.temperature
            },
        }

        if tools:
            req["toolConfig"] = {"tools": tools}

        return req