import json
import logging
import os
import socket
from typing import Literal, Optional
from urllib import error, request

from pydantic import SecretStr
from pydantic_settings import BaseSettings

from heracles_agents.exceptions import (
    LlmAuthenticationError,
    LlmBadRequestError,
    LlmConnectionError,
    LlmRateLimitError,
    LlmServiceUnavailableError,
    LlmTimeoutError,
    LlmUnknownError,
)

logger = logging.getLogger(__name__)


class HuggingFaceClientConfig(BaseSettings):
    """HTTP client for Hugging Face-compatible model servers.

    The server is expected to expose an OpenAI-compatible
    `/v1/chat/completions` endpoint, as provided by TGI, vLLM, or a small
    custom FastAPI wrapper.
    """

    client_type: Literal["huggingface"]
    host: Optional[str] = None
    api_key: Optional[SecretStr] = None
    timeout: int = 300
    max_new_tokens: int = 512
    top_p: Optional[float] = None
    seed: Optional[int] = None

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise NotImplementedError(
                "Only `text` format is currently implemented for Hugging Face servers"
            )
        if tools:
            raise NotImplementedError(
                "Native Hugging Face server tool calls are not implemented. Use "
                "`tool_interface: custom` or `tool_interface: none`."
            )

        payload = self._build_payload(model_info, messages)
        try:
            data = self._post_json(self._chat_completions_url(), payload)
            return [self._extract_message(data)]
        except (
            LlmAuthenticationError,
            LlmBadRequestError,
            LlmConnectionError,
            LlmRateLimitError,
            LlmServiceUnavailableError,
            LlmTimeoutError,
        ):
            raise
        except TimeoutError as ex:
            raise LlmTimeoutError(str(ex)) from ex
        except socket.timeout as ex:
            raise LlmTimeoutError(str(ex)) from ex
        except error.URLError as ex:
            raise LlmConnectionError(str(ex)) from ex
        except ValueError as ex:
            raise LlmBadRequestError(str(ex)) from ex
        except Exception as ex:
            raise LlmUnknownError(str(ex)) from ex

    def _build_payload(self, model_info, messages):
        payload = {
            "model": model_info.model,
            "messages": messages,
            "temperature": model_info.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": False,
        }
        seed = self.seed if self.seed is not None else getattr(model_info, "seed", None)
        if seed is not None:
            payload["seed"] = seed
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def _post_json(self, url, payload):
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as ex:
            self._raise_http_error(ex)

    def _chat_completions_url(self):
        host = self.host or os.environ.get("HUGGINGFACE_HOST", "http://localhost:8000")
        host = host.rstrip("/")
        if host.endswith("/v1"):
            return f"{host}/chat/completions"
        return f"{host}/v1/chat/completions"

    def _api_key_value(self):
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return os.environ.get("HUGGINGFACE_API_KEY")

    def _extract_message(self, data):
        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as ex:
            raise ValueError(f"Unexpected Hugging Face server response: {data}") from ex
        if content is None:
            content = ""
        return {"role": message.get("role", "assistant"), "content": content}

    def _raise_http_error(self, ex):
        body = ""
        try:
            body = ex.read().decode("utf-8")
        except Exception:
            pass
        msg = body or str(ex)
        if ex.code == 429:
            raise LlmRateLimitError(msg) from ex
        if ex.code == 408:
            raise LlmTimeoutError(msg) from ex
        if ex.code in {500, 502, 503, 504}:
            raise LlmServiceUnavailableError(msg) from ex
        if ex.code in {401, 403}:
            raise LlmAuthenticationError(msg) from ex
        if ex.code in {400, 404, 413, 422}:
            raise LlmBadRequestError(msg) from ex
        raise LlmUnknownError(msg) from ex
