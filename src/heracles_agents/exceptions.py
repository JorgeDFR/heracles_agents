class LlmProviderError(Exception):
    """Base class for all provider errors."""


class LlmRateLimitError(LlmProviderError):
    """Provider rate limit exceeded."""


class LlmTimeoutError(LlmProviderError):
    """Provider timeout."""


class LlmServiceUnavailableError(LlmProviderError):
    """Provider temporarily unavailable."""


class LlmAuthenticationError(LlmProviderError):
    """Authentication failure."""


class LlmConnectionError(LlmProviderError):
    """Network/transport failure."""


class LlmBadRequestError(LlmProviderError):
    """Malformed request."""


class LlmUnknownError(LlmProviderError):
    """Unexpected provider error."""