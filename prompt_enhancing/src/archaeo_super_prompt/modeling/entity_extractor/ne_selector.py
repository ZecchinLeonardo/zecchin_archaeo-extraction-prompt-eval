"""Module for abstract base Named Entities Selector class."""

from typing import cast
import pandas
from pandera.typing.pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .types import ChunksWithEntities, ChunksWithThesaurus, NamedEntityField
from . import model as ner_module


class NeSelector(ClassifierMixin, BaseEstimator, TransformerMixin):
    """Abstract class to select only chunks in which named thesaurus occur."""

    def __init__(
        self,
        to_extract: NamedEntityField,
        allowed_fuzzy_match_score=0.95,
    ) -> None:
        """Initialize the Named Entity Selector from the data about the field.

        Arguments:
            to_extract: the data about the field to be extracted from the \
entities to be filtered
            allowed_fuzzy_match_score: the treshold (between 0 and 1) above \
which a thesaurus match is kept
        """
        super().__init__()
        self._to_extract = to_extract
        self._allowed_fuzzy_match_score = allowed_fuzzy_match_score

    def transform(
        self, X: DataFrame[ChunksWithEntities]
    ) -> DataFrame[ChunksWithThesaurus]:
        """Filter the identified named entities and filter the chunks.

        According to the information about the field to be extracted, filter
        the named entities for each chunk and keep only chunks with a
        non-empty filtered named-entities list.
        """
        _, compatible_entities, thesaurus = self._to_extract
        chunk_contents = X["chunk_content"].to_list()
        entities = X["named_entities"].to_list()
        result = ner_module.extract_wanted_entities(
            chunk_contents,
            ner_module.filter_entities(entities, compatible_entities),
            thesaurus,
            self._allowed_fuzzy_match_score,
        )
        output = cast(pandas.DataFrame, X.copy().drop("named_entities"))
        output["identified_thesaurus"] = [
            list(r) if r is not None else None for r in result
        ]
        return ChunksWithThesaurus.validate(
            output[output["identified_thesaurus"].notnull()]
        )
