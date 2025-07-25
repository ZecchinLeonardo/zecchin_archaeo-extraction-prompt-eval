"""Management of the prompt attachment creation."""

from typing import cast
from pandera.typing.pandas import DataFrame
import pandas as pd
from sklearn.pipeline import FunctionTransformer
from ..entity_extractor.types import ChunksWithThesaurus
from .types import InputForExtractionWithSuggestedThesauri


def ChunksToText():
    """Unifies the filtered chunks into one attachment text for an LLM prompt.

    This pipeline Transformer applies this chunk merge for each intervention.
    """
    # TODO: define a unique ChunksWithSuggestedValues, regardless if its a
    # thesaurus identifier, an identified number, etc.

    def to_readable_context_string(
        filtered_chunks: DataFrame[ChunksWithThesaurus],
    ) -> str:
        msg: str = ""
        for _, chunk in filtered_chunks.sort_values(
            by="chunk_index"
        ).iterrows():
            msg += f"`%% {chunk['filename']} | Page {chunk['chunk_page_position']} ({chunk['chunk_type']}) %%`\n\n"
            msg += chunk["chunk_content"] + "\n" * 2
            msg += "`" + "-" * 60 + "`\n\n"
        return msg

    def unify_thesaurus(X: DataFrame[ChunksWithThesaurus]):
        return set().union(*X["identified_thesaurus"].tolist())

    def transform(
        X: DataFrame[ChunksWithThesaurus],
    ) -> DataFrame[InputForExtractionWithSuggestedThesauri]:
        return InputForExtractionWithSuggestedThesauri.validate(
            pd.DataFrame(
                (
                    lambda filtered_chunks: {
                        "id": id_,
                        "merged_chunks": to_readable_context_string(
                            filtered_chunks
                        ),
                        "identified_thesaurus": list(unify_thesaurus(
                            filtered_chunks
                        )),
                    }
                )(cast(DataFrame[ChunksWithThesaurus], filtered_chunks))
                for id_, filtered_chunks in X.groupby("id")
            )
        )

    return FunctionTransformer(transform)
