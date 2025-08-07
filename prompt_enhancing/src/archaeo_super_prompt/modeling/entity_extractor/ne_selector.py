"""Module for Named Entities Selector class with thesaurus-fuzzymatching."""

from typing import cast, override

import pandas
from pandera.typing.pandas import DataFrame
from tqdm import tqdm

from ...types.thesaurus import ThesaurusProvider
from ..types.base_transformer import BaseTransformer
from . import fuzzy_match
from .types import (
    ChunksWithEntities,
    ChunksWithThesaurus,
    CompleteEntity,
    NerXXLEntities,
)

# TODO: inherit it from a DetailedEvaluatorMixin when evaluation will be needed


class NeSelector(BaseTransformer):
    """Filter of chunks according to wanted strings among the entities."""

    def __init__(
        self,
        field_name: str,
        compatible_entities: set[NerXXLEntities],
        wanted_matches: ThesaurusProvider,
        keep_chunks_without_identified_values=False,
    ):
        """Initialize the Named Entity Selector from the data about the field.

        Arguments:
            field_name: a label describing the entities to be extracted
            compatible_entities: a set of entity types to consider for selecting the chunks
            wanted_matches: a frozen function giving at runtime the list of matches (can be huge)
            keep_chunks_without_identified_values: if True, the chunks with \
entities in the desired group of entity types are always kept, even if no \
thesaurus has been identified among these entities. If False, these chunks \
are only kept if there is not any chunk where hesaurus has been identified.

        Return:
        A Transformer to select only chunks in which named thesaurus occur.
        """
        self.field_name = field_name
        self.compatible_entities = compatible_entities
        self.wanted_matches = wanted_matches
        self.keep_chunks_without_identified_values = (
            keep_chunks_without_identified_values
        )

    @override
    def transform(
        self,
        X: DataFrame[ChunksWithEntities],
    ) -> DataFrame[ChunksWithThesaurus]:
        """Filter the identified named entities and filter the chunks.

        According to the information about the field to be extracted, filter
        the named entities for each chunk and keep only chunks with a
        non-empty filtered named-entities list.
        """
        chunk_contents = (cast(str, r.chunk_content) for r in X.itertuples())
        entities = X["named_entities"].to_list()
        result = fuzzy_match.extract_wanted_entities(
            chunk_contents,
            (
                [
                    entity
                    for entity in cast(list[CompleteEntity], entity_list)
                    if entity.entity in self.compatible_entities
                ]
                for entity_list in entities
            ),
            self.wanted_matches,
        )
        output = cast(
            pandas.DataFrame, X.copy().drop(columns="named_entities")
        )
        output["identified_thesaurus"] = [
            list(r) if r is not None else None
            for r in tqdm(
                result,
                total=len(X),
                desc="Fuzzy-search thesaurus in text chunks.",
                unit="analyzed chunk",
            )
        ]
        filtered_chunks = ChunksWithThesaurus.validate(
            output[output["identified_thesaurus"].notnull()]
        )
        # keep only the chunks with thesaurus if needed
        if not self.keep_chunks_without_identified_values:
            return filtered_chunks
        chunks_with_thesaurus = filtered_chunks[
            filtered_chunks["identified_thesaurus"].apply(
                lambda lst: len(lst) > 0
            )
        ]
        if len(chunks_with_thesaurus) > 0:
            return chunks_with_thesaurus
        return filtered_chunks
