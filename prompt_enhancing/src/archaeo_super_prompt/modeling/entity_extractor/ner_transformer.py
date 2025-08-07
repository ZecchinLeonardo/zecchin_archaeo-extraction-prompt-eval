"""The pipeline Transformer related to the remote NER model."""

from typing import cast, override

import pandas as pd
from pandera.typing.pandas import DataFrame

from ...types.pdfchunks import PDFChunkDataset
from ..types.base_transformer import BaseTransformer
from . import model as ner_module
from .types import EntitiesPerChunkSchema


class NerModel(BaseTransformer):
    """Transformer adding identified NamedRecognition features for each chunk."""

    def __init__(
        self,
        allowed_ner_confidence=0.70,
    ):
        """Instantiate the Named Entity Recognition model.

        Environment variables:
            The NER_MODEL_HOST_URL env var must be set with the base url of the
            remote model for the named entity recognition (e.g.
            'http://localhost:8004')
        """
        self.allowed_ner_confidence = allowed_ner_confidence

    @override
    def transform(
        self,
        X: PDFChunkDataset,
    ) -> DataFrame[EntitiesPerChunkSchema]:
        chunk_contents = list(
            map(lambda row: cast(str, row.chunk_content), X.itertuples())
        )
        result = ner_module.fetch_entities(chunk_contents)
        result = ner_module.postrocess_entities(
            result, self.allowed_ner_confidence
        )
        return EntitiesPerChunkSchema.validate(
            pd.DataFrame([{"named_entities": lst} for lst in result])
        )
