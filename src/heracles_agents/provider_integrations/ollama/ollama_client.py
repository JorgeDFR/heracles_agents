import os
import logging

from typing import Literal
from ollama import ChatResponse, RequestError, ResponseError, chat, Client

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
from heracles_agents.inference_parameters import get_reasoning_settings

logger = logging.getLogger(__name__)


class OllamaClientConfig(BaseSettings):
    client_type: Literal["ollama"]
    keep_alive: str | int | None = None
    _chat_func: object = PrivateAttr(default=None)


    def __init__(self, **data):
        super().__init__(**data)
        ollama_host = os.environ.get("OLLAMA_HOST", "https://ollama.com")
        api_key = os.environ.get("OLLAMA_API_KEY")

        if ollama_host == "https://ollama.com" and api_key:
            client = Client(
                host=ollama_host,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            self._chat_func = client.chat
        else:
            self._chat_func = chat


    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise NotImplementedError(
                "Only `text` format is currently implemented for interfacing with Ollama"
            )

        options = {}

        if getattr(model_info, "temperature", None) is not None:
            options["temperature"] = model_info.temperature

        if model_info.seed is not None:
            options["seed"] = model_info.seed

        try:
            request = {
                "model": model_info.model,
                "messages": messages,
                "tools": tools,
                "options": options,
            }
            reasoning_mode, reasoning_effort = get_reasoning_settings(model_info)
            if reasoning_mode == "enabled":
                request["think"] = reasoning_effort or True
            elif reasoning_mode == "disabled":
                request["think"] = False
            elif reasoning_mode is None:
                # Preserve the behavior of legacy experiment files that do not
                # yet contain normalized reasoning settings.
                request["think"] = False
            if self.keep_alive is not None:
                request["keep_alive"] = self.keep_alive
            return self._chat_func(
                **request,
            )

        except TimeoutError as ex:
            raise LlmTimeoutError(str(ex)) from ex

        except ConnectionError as ex:
            raise LlmConnectionError(str(ex)) from ex

        except RequestError as ex:
            raise LlmConnectionError(str(ex)) from ex

        except ResponseError as ex:
            self._raise_response_error(ex)

        except ValueError as ex:
            raise LlmBadRequestError(str(ex)) from ex

        except Exception as ex:
            raise LlmUnknownError(str(ex)) from ex


    def _raise_response_error(self, ex: ResponseError):
        if ex.status_code == 429:
            raise LlmRateLimitError(str(ex)) from ex
        if ex.status_code == 408:
            raise LlmTimeoutError(str(ex)) from ex
        if ex.status_code in {500, 502, 503, 504}:
            raise LlmServiceUnavailableError(str(ex)) from ex
        if ex.status_code in {401, 403}:
            raise LlmAuthenticationError(str(ex)) from ex
        if ex.status_code in {400, 404, 413, 422}:
            raise LlmBadRequestError(str(ex)) from ex
        raise LlmUnknownError(str(ex)) from ex
