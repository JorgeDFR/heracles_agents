from pydantic import BaseModel


class StructuredToolDescription(BaseModel):
    """Description of a structured tool / function"""

    name: str
    description: str
    grammar: str
