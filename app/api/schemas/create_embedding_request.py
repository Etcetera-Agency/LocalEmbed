from pydantic import BaseModel
from typing import List, Literal

InputType = Literal["passage", "query"]


class CreateEmbeddingRequest(BaseModel):
    input: str | List[str]
    """Input text to embed, encoded as a string or array of strings."""

    model: str
    """ID of the model to use. See the /v1/models endpoint for a list of available models."""

    input_type: InputType | None = None
    """Optional E5 input type. Use passage for stored text and query for search queries."""

    # dimensions: int | None = None
    # TODO: """The number of dimensions the resulting output embeddings should have."""
