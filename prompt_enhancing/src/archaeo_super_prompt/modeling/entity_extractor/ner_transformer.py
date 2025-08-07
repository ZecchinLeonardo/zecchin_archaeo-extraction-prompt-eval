"""The pipeline Transformer related to the remote NER model."""

from typing import cast
from pandera.typing.pandas import DataFrame
import pandas as pd
from sklearn.pipeline import FunctionTransformer

from . import model as ner_module
from ...types.pdfchunks import PDFChunkDataset
from .types import EntitiesPerChunkSchema


def NerModel(
    allowed_ner_confidence=0.70,
):
    """Transformer adding identified NamedRecognition features for each chunk."""

    def NERecognize(
        X: PDFChunkDataset,
    ) -> DataFrame[EntitiesPerChunkSchema]:
        chunk_contents = list(
            map(lambda row: cast(str, row.chunk_content), X.itertuples())
        )
        result = ner_module.fetch_entities(chunk_contents)
        result = ner_module.postrocess_entities(result, allowed_ner_confidence)
        return EntitiesPerChunkSchema.validate(
            pd.DataFrame([{"named_entities": lst} for lst in result])
        )

    return FunctionTransformer(NERecognize)
