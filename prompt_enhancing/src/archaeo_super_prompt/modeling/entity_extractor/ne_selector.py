"""Module for Named Entities Selector class with thesaurus-fuzzymatching."""

from typing import cast
import pandas
from pandera.typing.pandas import DataFrame
from sklearn.pipeline import FunctionTransformer
from tqdm import tqdm

from .types import (
    ChunksWithEntities,
    ChunksWithThesaurus,
    CompleteEntity,
    NamedEntityField,
)
from . import fuzzy_match

# TODO: inherit it from a DetailedEvaluatorMixin when evaluation will be needed


def NeSelector(
    to_extract: NamedEntityField,
    keep_chunks_without_identified_thesaurus=False,
):
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

    Return:
    A Transformer to select only chunks in which named thesaurus occur.
    """

    def transform(
        X: DataFrame[ChunksWithEntities],
    ) -> DataFrame[ChunksWithThesaurus]:
        """Filter the identified named entities and filter the chunks.

        According to the information about the field to be extracted, filter
        the named entities for each chunk and keep only chunks with a
        non-empty filtered named-entities list.
        """
        _, compatible_entities, thesaurus = to_extract
        chunk_contents = (cast(str, r.chunk_content) for r in X.itertuples())
        entities = X["named_entities"].to_list()
        result = fuzzy_match.extract_wanted_entities(
            chunk_contents,
            (
                [
                    entity
                    for entity in cast(list[CompleteEntity], entity_list)
                    if entity.entity in compatible_entities
                ]
                for entity_list in entities
            ),
            thesaurus,
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
        if not keep_chunks_without_identified_thesaurus:
            return filtered_chunks
        chunks_with_thesaurus = filtered_chunks[
            filtered_chunks["identified_thesaurus"].apply(
                lambda lst: len(lst) > 0
            )
        ]
        if len(chunks_with_thesaurus) > 0:
            return chunks_with_thesaurus
        return filtered_chunks

    return FunctionTransformer(transform)
