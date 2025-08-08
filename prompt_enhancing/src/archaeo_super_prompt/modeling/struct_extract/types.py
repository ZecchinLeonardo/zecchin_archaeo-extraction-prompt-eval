"""Types for structured data extraction models."""

from typing import NamedTuple
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import Series, Index


class BaseInputForExtraction(DataFrameModel):
    """Uniformized base input struct for a FieldExtractor.

    The Knowledge Generic type must be a TypedDict. It represents contextual
    metadata which is known or is supposed to be True after an inference before
    another prior model.
    """

    id: Index[int]
    merged_chunks: Series[str]


class BaseInputForExtractionRowSchema(NamedTuple):
    """Base Schema for typesafely iterating over rows."""

    Index: int
    merged_chunks: str


class InputForExtractionWithSuggestedThesauri(BaseInputForExtraction):
    """For each intervention, a list of thesarus identifiers is given."""

    identified_thesaurus: list[int]


class InputForExtractionWithSuggestedThesauriRowSchema(
    BaseInputForExtractionRowSchema
):
    """For each intervention, a list of thesarus identifiers is given."""

    identified_thesaurus: list[int]
