from typing import Any, Literal

from pydantic import BaseModel


MessageKind = Literal[
    "assistant_text",
    "tool_call",
    "answer_tool",
    "reasoning",
    "tool_result",
    "unknown",
]


class NormalizedMessage(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    kind: MessageKind
    text: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_id: str | None = None
    raw: Any = None


def normalized_summary(message: NormalizedMessage):
    match message.kind:
        case "tool_call":
            args = ",".join(
                f"{key}={value}" for key, value in (message.tool_args or {}).items()
            )
            return f"Function Call: {message.tool_name}({args},)"
        case "tool_result":
            return f"Tool result: {message.text}"
        case _:
            return message.text
