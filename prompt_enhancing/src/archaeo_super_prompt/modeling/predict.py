"""Code containing the global model and a way to infer into it."""
from pandera.typing.pandas import DataFrame
from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion
import sklearn


from .pdf_to_text import VLLM_Preprocessing
from ..dataset.thesauri import load_comune
from .entity_extractor import NamedEntityField, NerModel, NeSelector
from .struct_extract.chunks_to_text import ChunksToText
from .struct_extract.extractors.comune import ComuneExtractor
from .struct_extract.language_model import get_vllm_model

def identity():
    """Set an identity pipeline Transformer."""

    def IdentityFunction(X: DataFrame) -> DataFrame:
        return X

    return FunctionTransformer(IdentityFunction)


def get_pipeline():
    """Return the main pipeline as a Directed Acyclic Graph."""
    llm_model = get_vllm_model(temperature=0.05)
    with sklearn.config_context(transform_output="pandas"):
        return Pipeline(
            [
                (
                    "vllm",
                    VLLM_Preprocessing(
                        model="granite3.2-vision:latest",
                        incipit_only=True,
                        prompt="OCR this part of Italian document for markdown-based processing.",
                        embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
                    ),
                ),
                (
                    "ner",
                    FeatureUnion(
                        [
                            ("identity", identity()),
                            ("ner", NerModel()),
                        ]
                    ).set_output(transform="pandas"),
                ),
                (
                    "sel-comune",
                    NeSelector(
                        NamedEntityField(
                            "comune",
                            {
                                "INDIRIZZO",
                                "CODICE_POSTALE",
                                "LUOGO",
                            },
                            load_comune,
                        )
                    ),
                ),
                ("merge", ChunksToText()),
                ("extract-comune", ComuneExtractor(llm_model))
            ],
        )
