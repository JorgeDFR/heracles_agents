from pydantic import BaseModel, PrivateAttr


class StructuredToolDescription(BaseModel):
    """Description of a structured tool / function"""

    name: str
    description: str
    grammar: str
    _bound_args: PrivateAttr() = None
