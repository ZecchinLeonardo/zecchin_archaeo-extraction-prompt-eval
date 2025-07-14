from typing import List, NamedTuple, Set, cast
from collections.abc import Callable
from pandera.typing.pandas import DataFrame
import pandas as pd
from sklearn.pipeline import FunctionTransformer

from ...types.ner_labeled_chunks import NerLabeledChunkDatasetSchema

from . import model as ner_module
from .types import NerXXLEntities
from ...types.pdfchunks import PDFChunkDataset


class NamedEntityField(NamedTuple):
    name: str
    compatible_entities: set[NerXXLEntities]
    thesaurus_values: Callable[[], list[str]]


def NerModel(
    to_extract: list[NamedEntityField],
    allowed_ner_confidence=0.70,
    allowed_fuzzy_match_score=0.70,
):
    def transform(
        X: PDFChunkDataset,
    ) -> DataFrame[NerLabeledChunkDatasetSchema]:
        result = ner_module.fetch_entities(
            list(map(lambda row: cast(str, row.chunk_content), X.itertuples()))
        )
        result = ner_module.postrocess_entities(result, allowed_ner_confidence)
        result = {
            name: ner_module.extract_wanted_entities(
                ner_module.filter_entities(result, compatible_entities),
                thesaurus,
                allowed_fuzzy_match_score,
            )
            for name, compatible_entities, thesaurus in to_extract
        }
        return NerLabeledChunkDatasetSchema.validate(
            pd.DataFrame(
                [
                    {
                        "nerIdentifiedThesaurus": {
                            k: result[k][i]
                            for k in result
                            if result[k][i] is not None
                        }
                    }
                    for i in range(len(X))
                ]
            )
        )

    return FunctionTransformer(transform)
