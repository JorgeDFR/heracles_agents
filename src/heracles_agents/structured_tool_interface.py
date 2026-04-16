from pydantic import BaseModel


class StructuredToolDescription(BaseModel):
    """Description of a structured tool / function"""

    name: str
    description: str
    grammar: str

    def to_openai_responses(self):
        t = {
            "type": "custom",
            "name": self.name,
            "description": self.description,
            "format": {"type": "grammar", "syntax": "lark", "definition": self.grammar},
        }
        return t

    def to_anthropic(self):
        raise NotImplementedError(
            "Structured tool calling not supported for Anthropic tools"
        )

    def to_ollama(self):
        raise NotImplementedError(
            "Structured tool calling not supported for Ollama tools"
        )

    def to_bedrock(self):
        raise NotImplementedError(
            "Structured tool calling not supported for Bedrock tools"
        )

    def to_openrouter(self):
        raise NotImplementedError(
            "Structured tool calling not supported for OpenRouter tools"
        )

    def to_custom(self):
        raise NotImplementedError(
            "Structured tool calling not supported for custom tools"
        )
