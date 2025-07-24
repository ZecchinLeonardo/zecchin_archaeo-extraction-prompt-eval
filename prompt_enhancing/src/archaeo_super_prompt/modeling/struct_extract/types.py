"""Types for structured data extraction models."""

from typing import NamedTuple, TypedDict
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import Series


class BaseKnowledgeDataScheme(TypedDict):
    """Base dictionary representing the knowledge for biasing an extraction task."""

    pass


class InputForExtraction(DataFrameModel):
    """Uniformized base input struct for a FieldExtractor.

    The Knowledge Generic type must be a TypedDict. It represents contextual
    metadata which is known or is supposed to be True after an inference before
    another prior model.
    """

    id: Series[int]
    merged_chunks: Series[str]
    knowledge: BaseKnowledgeDataScheme


class InputForExtractionRowSchema(NamedTuple):
    """Base Schema for typesafely iterating over rows."""

    id: int
    merged_chunks: str
    knowledge: BaseKnowledgeDataScheme
