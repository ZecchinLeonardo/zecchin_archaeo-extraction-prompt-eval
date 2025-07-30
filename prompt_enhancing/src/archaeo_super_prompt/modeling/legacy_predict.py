"""Legacy model for comparison."""

from pandera.typing.pandas import DataFrame
from sklearn.pipeline import FunctionTransformer, Pipeline
import sklearn

from .entity_extractor.types import (
    ChunksWithThesaurus,
)
from .struct_extract.language_model import get_vllm_model

from ..types.pdfchunks import PDFChunkDatasetSchema


from .pdf_to_text import VLLM_Preprocessing
from .struct_extract.legacy_extractor.main_transformer import MagohDataExtractor


def add_empty_suggested_thesaurus_list():
    """To fit data schema."""

    def transform(
        X: DataFrame[PDFChunkDatasetSchema],
    ) -> DataFrame[ChunksWithThesaurus]:
        X["identified_thesaurus"] = [[] for _ in range(len(X))]
        return ChunksWithThesaurus.validate(X)

    return FunctionTransformer(transform)


def get_legacy_model():
    """Return the legacy model but with the vllm as pre-processing layer."""
    llm_model = get_vllm_model(temperature=0.05)
    with sklearn.config_context(transform_output="pandas"):
        return Pipeline(
            [
                (
                    "vllm",
                    VLLM_Preprocessing(
                        model="granite3.2-vision:latest",
                        prompt="OCR this part of Italian document for markdown-based processing.",
                        embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
                        incipit_only=True,
                    ),
                ),
                ("extractor", MagohDataExtractor(llm_model)),
            ],
        )
