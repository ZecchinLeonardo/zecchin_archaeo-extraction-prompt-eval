"""Types for structured data extraction models."""

from collections.abc import Iterator
from typing import NamedTuple, cast
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import DataFrame, Series


class InputForExtraction[T](DataFrameModel):
    """Uniformized input struct for a FieldExtractor."""

    id: Series[int]
    merged_chunks: Series[str]
    suggested_extraction_outputs: list[T]


class InputForExtractionRowSchema[T](NamedTuple):
    """Schema for typesafely iterating over rows."""

    id: int
    merged_chunks: str
    suggested_extraction_outputs: list[T]


def itertuples[T](
    df: DataFrame[InputForExtraction[T]],
) -> Iterator[InputForExtractionRowSchema[T]]:
    """Typesafe wrapper for itertuples."""
    return cast(Iterator[InputForExtractionRowSchema[T]], df.itertuples())
