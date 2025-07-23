"""Types for structured data extraction models."""

from collections.abc import Iterator
from typing import NamedTuple, TypedDict, cast
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import DataFrame, Series

class BaseKnowledgeDataScheme(TypedDict):
    """Base dictionary representing the knowledge for biasing an extraction task."""
    pass

class InputForExtraction[KnowledgeDataDict: BaseKnowledgeDataScheme](DataFrameModel):
    """Uniformized input struct for a FieldExtractor.

    The Knowledge Generic type must be a TypedDict. It represents contextual
    metadata which is known or is supposed to be True after an inference before
    another prior model.

    Type Parameters:
        KnowledgeDataDict: a dictionary type which bring already inferred \
knowledge information that can be used to skew the extraction model.
    """

    id: Series[int]
    merged_chunks: Series[str]
    knowledge: KnowledgeDataDict


class InputForExtractionRowSchema[KnowledgeDataDict: BaseKnowledgeDataScheme](NamedTuple):
    """Schema for typesafely iterating over rows."""

    id: int
    merged_chunks: str
    knowledge: KnowledgeDataDict


def itertuples[KnowledgeDataDict: BaseKnowledgeDataScheme](
    df: DataFrame[InputForExtraction[KnowledgeDataDict]],
) -> Iterator[InputForExtractionRowSchema[KnowledgeDataDict]]:
    """Typesafe wrapper for itertuples."""
    return cast(Iterator[InputForExtractionRowSchema[KnowledgeDataDict]], df.itertuples())
