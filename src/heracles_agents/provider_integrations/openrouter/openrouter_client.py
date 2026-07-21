from typing import Literal

import logging
import time

import httpx
from openrouter import OpenRouter
from openrouter.errors import (
    TooManyRequestsResponseError,
    RequestTimeoutResponseError,
    EdgeNetworkTimeoutResponseError,
    ServiceUnavailableResponseError,
    ProviderOverloadedResponseError,
    InternalServerResponseError,
    BadGatewayResponseError,
    NoResponseError,
    UnauthorizedResponseError,
    ForbiddenResponseError,
    BadRequestResponseError,
    UnprocessableEntityResponseError,
    PayloadTooLargeResponseError,
    NotFoundResponseError,
    ConflictResponseError,
    OpenRouterError,
)

from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings

from heracles_agents.exceptions import (
    LlmRateLimitError,
    LlmTimeoutError,
    LlmServiceUnavailableError,
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmUnknownError,
)

logger = logging.getLogger(__name__)


class OpenRouterClientConfig(BaseSettings):
    client_type: Literal["openrouter"]
    auth_key: SecretStr = Field(alias="HERACLES_OPENROUTER_API_KEY", exclude=True)
    _client: OpenRouter = PrivateAttr()
    _model_pricing_cache: dict = PrivateAttr(default_factory=dict)


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

    def get_generation_stats(
        self,
        generation_id: str,
        *,
        attempts: int = 4,
        wait_time_s: float = 0.5,
    ) -> dict | None:
        headers = {
            "Authorization": f"Bearer {self.auth_key.get_secret_value()}",
        }
        last_exception = None
        for idx in range(attempts):
            try:
                response = httpx.get(
                    "https://openrouter.ai/api/v1/generation",
                    params={"id": generation_id},
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, dict):
                    return data
            except Exception as ex:
                last_exception = ex

            if idx < attempts - 1:
                time.sleep(wait_time_s)

        logger.warning(
            "Could not fetch OpenRouter generation stats for %s: %s",
            generation_id,
            last_exception,
        )
        return None

    def get_model_pricing(self, model_identifier: str) -> dict | None:
        if model_identifier in self._model_pricing_cache:
            return self._model_pricing_cache[model_identifier]

        try:
            response = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "Authorization": f"Bearer {self.auth_key.get_secret_value()}",
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                return None
            for model in models:
                if not isinstance(model, dict):
                    continue
                if model.get("id") != model_identifier:
                    continue
                pricing = model.get("pricing")
                if isinstance(pricing, dict):
                    self._model_pricing_cache[model_identifier] = pricing
                    return pricing
        except Exception as ex:
            logger.warning(
                "Could not fetch OpenRouter model pricing for %s: %s",
                model_identifier,
                ex,
            )
        return None


    def _build_payload(self, model_info, tools, messages):
        payload = {
            "model": model_info.model,
            "messages": messages,
            "tools": tools,
            "temperature": model_info.temperature,
            "x_open_router_metadata": "enabled",
        }

        if getattr(model_info, "seed", None) is not None:
            payload["seed"] = model_info.seed

        if getattr(model_info, "reasoning", None) is not None:
            payload["reasoning"] = {
                "effort": model_info.reasoning
            }

        return payload


    def _raise_normalized_error(self, ex: Exception):
        # ----------------------------------------------------------
        # Rate limiting
        # ----------------------------------------------------------
        if isinstance(ex, TooManyRequestsResponseError):
            raise LlmRateLimitError(str(ex)) from ex

        # ----------------------------------------------------------
        # Timeouts
        # ----------------------------------------------------------
        if isinstance(
            ex,
            (
                RequestTimeoutResponseError,
                EdgeNetworkTimeoutResponseError,
            ),
        ):
            raise LlmTimeoutError(str(ex)) from ex

        # ----------------------------------------------------------
        # Temporary / retryable provider failures
        # ----------------------------------------------------------
        if isinstance(
            ex,
            (
                ServiceUnavailableResponseError,
                ProviderOverloadedResponseError,
                InternalServerResponseError,
                BadGatewayResponseError,
                NoResponseError,
            ),
        ):
            raise LlmServiceUnavailableError(str(ex)) from ex

        # ----------------------------------------------------------
        # Authentication / authorization
        # ----------------------------------------------------------
        if isinstance(
            ex,
            (
                UnauthorizedResponseError,
                ForbiddenResponseError,
            ),
        ):
            raise LlmAuthenticationError(str(ex)) from ex

        # ----------------------------------------------------------
        # Client-side request problems
        # ----------------------------------------------------------
        if isinstance(
            ex,
            (
                BadRequestResponseError,
                UnprocessableEntityResponseError,
                PayloadTooLargeResponseError,
                NotFoundResponseError,
                ConflictResponseError,
            ),
        ):
            raise LlmBadRequestError(str(ex)) from ex

        # ----------------------------------------------------------
        # Generic OpenRouter SDK error
        # ----------------------------------------------------------
        if isinstance(ex, OpenRouterError):
            raise LlmUnknownError(str(ex)) from ex

        # ----------------------------------------------------------
        # Unknown fallback
        # ----------------------------------------------------------
        raise LlmUnknownError(str(ex)) from ex
