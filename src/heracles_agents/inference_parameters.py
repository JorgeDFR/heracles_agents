"""Provider-neutral inference parameter models and adapters."""

from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class ReasoningSettings(BaseModel):
    """Provider-neutral reasoning intent for one benchmark model.

    ``unsupported`` records that a model has no reasoning control, while
    ``provider_default`` deliberately leaves the setting uncontrolled. Both
    modes omit the provider request parameter, but retain different provenance.
    """

    mode: Literal["enabled", "disabled", "unsupported", "provider_default"]
    effort: Optional[str] = None

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, value):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("reasoning effort must be a non-empty string or null")
        return value.strip()

    @model_validator(mode="after")
    def validate_mode_and_effort(self):
        if self.mode != "enabled" and self.effort is not None:
            raise ValueError(
                "reasoning effort may only be set when reasoning mode is enabled"
            )
        return self


def get_reasoning_settings(model_info) -> tuple[str | None, str | None]:
    """Read normalized settings while tolerating legacy/simple test objects."""

    value = getattr(model_info, "reasoning", None)
    if value is None:
        return None, None
    if isinstance(value, str):
        return "enabled", value
    if isinstance(value, bool):
        return ("enabled" if value else "disabled"), None
    if isinstance(value, dict):
        return value.get("mode"), value.get("effort")
    return getattr(value, "mode", None), getattr(value, "effort", None)
