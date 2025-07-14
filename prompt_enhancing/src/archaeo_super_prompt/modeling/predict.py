from sklearn.pipeline import FunctionTransformer, Pipeline, FeatureUnion

from archaeo_super_prompt.types.pdfchunks import PDFChunkDataset
from .pdf_to_text import VLLM_Preprocessing
from ..dataset.thesaurus import load_comune
from .entity_extractor import NamedEntityField, NerModel


def identity():
    def transform(X: PDFChunkDataset):
        return X
    return FunctionTransformer(transform)

def get_pipeline():
    return Pipeline(
        [
            (
                "vllm",
                VLLM_Preprocessing(
                    model="granite3.2-vision:latest",
                    prompt="OCR this part of Italian document for markdown-based processing.",
                    embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
                ),
            ),
            (
                "pre_selection",
                FeatureUnion(
                    [
                        ("identity", identity()),
                        (
                            "ner",
                            NerModel(
                                [
                                    NamedEntityField(
                                        "comune",
                                        {
                                            "INDIRIZZO",
                                            "CODICE_POSTALE",
                                            "LUOGO",
                                        },
                                        load_comune,
                                    )
                                ]
                            ),
                        ),
                        # TODO: add here the semantical pre-selection model
                    ]
                ).set_output(transform="pandas"),
            ),
        ]
    )
