from typing import Dict, List, Set, cast

from pandera.typing.pandas import DataFrame
import pandas as pd
from sklearn.pipeline import FunctionTransformer

from ...types.ner_labeled_chunks import NerLabeledChunkDatasetSchema

from . import model as ner_module
from .types import CompleteEntity, NerXXLEntities
from ...types.pdfchunks import PDFChunkDataset


def NerModel(to_extract: Dict[str, Set[NerXXLEntities]], allowed_confidence=0.70):
    # TODO: load the thesaurus
    thesaurus = {}

    def identify_thesaurus(entities: List[CompleteEntity], thesaurus: List[str]):
        # TODO:
        return []

    def transform(X: PDFChunkDataset) -> DataFrame[NerLabeledChunkDatasetSchema]:
        result = ner_module.fetch_entities(
            list(map(lambda row: cast(str, row.chunk_content), X.itertuples()))
        )
        result = ner_module.postrocess_entities(result, allowed_confidence)
        result = {
            field_key: ner_module.filter_entities(result, to_extract[field_key])
            for field_key in to_extract
        }
        return NerLabeledChunkDatasetSchema.validate(
            pd.DataFrame(
                [
                    {
                        "nerIdentifiedThesaurus": {
                            k: identify_thesaurus(result[k][i], thesaurus[k])
                            for k in result
                        }
                    }
                    for i in range(len(X))
                ]
            )
        )

    return FunctionTransformer(transform)
