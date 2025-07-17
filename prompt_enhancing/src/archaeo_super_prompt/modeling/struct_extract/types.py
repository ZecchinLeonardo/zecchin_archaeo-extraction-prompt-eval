"""Types for structured data extraction models."""

from pandera.pandas import DataFrameModel
from pandera.typing.pandas import Series


class InputForExtraction[T](DataFrameModel):
    """Uniformized input struct for a FieldExtractor."""

    id: Series[int]
    merged_chunks: Series[str]
    suggested_extraction_outputs: list[T]
