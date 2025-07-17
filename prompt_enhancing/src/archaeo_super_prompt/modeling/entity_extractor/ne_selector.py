"""Module for abstract base Named Entities Selector class."""

from typing import cast
import pandas
from pandera.typing.pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .types import ChunksWithEntities, ChunksWithThesaurus, NamedEntityField
from . import model as ner_module

# TODO: inherit it from a DetailedEvaluatorMixin when evaluation will be needed

class NeSelector(ClassifierMixin, BaseEstimator, TransformerMixin):
    """A Transformer to select only chunks in which named thesaurus occur."""

    def __init__(
        self,
        to_extract: NamedEntityField,
        keep_chunks_without_identified_thesaurus=False,
        allowed_fuzzy_match_score=0.95,
    ) -> None:
        """Initialize the Named Entity Selector from the data about the field.

        Arguments:
            to_extract: the data about the field to be extracted from the \
entities to be filtered
            keep_chunks_without_identified_thesaurus: if True, the chunks with \
entities in the desired group of entity types are always kept, even if no \
thesaurus has been identified among these entities. If False, these chunks are \
only kept if there is not any chunk where thesaurus has been identified.
            allowed_fuzzy_match_score: the treshold (between 0 and 1) above \
which a thesaurus match is kept
        """
        super().__init__()
        self._to_extract = to_extract
        self._allowed_fuzzy_match_score = allowed_fuzzy_match_score
        self._keep_chunks_without_identified_thesaurus = (
            keep_chunks_without_identified_thesaurus
        )

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
        output = cast(pandas.DataFrame, X.copy().drop(columns="named_entities"))
        output["identified_thesaurus"] = [
            list(r) if r is not None else None for r in result
        ]
        filtered_chunks = ChunksWithThesaurus.validate(
            output[output["identified_thesaurus"].notnull()]
        )
        # keep only the chunks with thesaurus if needed
        if self._keep_chunks_without_identified_thesaurus:
            return filtered_chunks
        chunks_with_thesaurus = filtered_chunks[
            filtered_chunks["identified_thesaurus"].apply(
                lambda lst: len(lst) > 0
            )
        ]
        if len(chunks_with_thesaurus) > 0:
            return chunks_with_thesaurus
        return filtered_chunks
