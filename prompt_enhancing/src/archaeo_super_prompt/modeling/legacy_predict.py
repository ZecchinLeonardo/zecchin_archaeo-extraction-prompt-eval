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


def get_legacy_model():
    """Return the legacy model but with the vllm as pre-processing layer."""
    llm_model = get_vllm_model(temperature=0.05)
    with sklearn.config_context(transform_output="pandas"):
        return Pipeline(
            [
                (
                    "vllm",
                    VLLM_Preprocessing(
                        vlm_provider='vllm',
                        vlm_model_id="ibm-granite/granite-vision-3.3-2b",
                        prompt="OCR this part of Italian document for markdown-based processing.",
                        embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
                        incipit_only=True,
                    ),
                ),
                ("extractor", MagohDataExtractor(llm_model)),
            ],
        )
