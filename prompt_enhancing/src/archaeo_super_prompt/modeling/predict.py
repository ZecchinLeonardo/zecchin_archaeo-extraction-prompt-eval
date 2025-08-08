"""Code containing the global model and a way to infer into it."""

import sklearn
from pandera.typing.pandas import DataFrame
from skdag import DAGBuilder
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline

from ..dataset.thesauri import load_comune
from .entity_extractor import NerModel, NeSelector
from .pdf_to_text import VLLM_Preprocessing
from .struct_extract.chunks_to_text import ChunksToText
from .struct_extract.extractors.archiving_date import ArchivingDateProvider
from .struct_extract.extractors.comune import ComuneExtractor
from .struct_extract.extractors.intervention_date import (
    InterventionStartExtractor,
)


def identity():
    """Set an identity pipeline Transformer."""

    def IdentityFunction(X: DataFrame) -> DataFrame:
        return X

    return FunctionTransformer(IdentityFunction)


def get_dag():
    """Return the main pipeline as a Directed Acyclic Graph.

    This structure allows to show the conditional dependencies between the
    inferences of the fields.
    """
    llm_model_id = "google/gemma-3-27b-it"
    llm_provider = "vllm"
    llm_model_temp = 0.05

    with sklearn.config_context(transform_output="pandas"):
        return (
            DAGBuilder(infer_dataframe=True)
            .add_step(
                "vllm",
                VLLM_Preprocessing(
                    vlm_provider="vllm",
                    vlm_model_id="ibm-granite/granite-vision-3.3-2b",
                    incipit_only=True,
                    prompt="OCR this part of Italian document for markdown-based processing.",
                    embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
                ),
            )
            .add_step("ner", NerModel(), deps=["vllm"])
            .add_step("ner-featured", "passthrough", deps=["vllm", "ner"])
            .add_step("archiving-date", ArchivingDateProvider(), deps=["vllm"])
            .add_step(
                "sel-comune",
                NeSelector(
                    "comune",
                    {
                        "INDIRIZZO",
                        "CODICE_POSTALE",
                        "LUOGO",
                    },
                    load_comune,
                ),
                deps=["ner-featured"],
            )
            .add_step("merge-comune", ChunksToText(), deps=["sel-comune"])
            # .add_step(
            #    "comune-extraction",
            #    ComuneExtractor(llm_provider, llm_model_id, llm_model_temp),
            #    deps=["merge-comune"],
            # )
            .add_step(
                "date-chunk-filter",
                Pipeline(
                    [
                        (
                            "sel-date",
                            NeSelector(
                                "data",
                                {
                                    "DATA",
                                },
                                lambda: list(
                                    enumerate(
                                        [
                                            "primavera",
                                            "estate",
                                            "autunno",
                                            "inverno",
                                        ]
                                    )
                                ),
                                True,
                            ),
                        ),
                        ("merge", ChunksToText()),
                    ]
                ),
                deps=["ner-featured"],
            )
            .add_step(
                "intervention-start-extraction",
                InterventionStartExtractor(
                    llm_provider, llm_model_id, llm_model_temp
                ),
                deps=["date-chunk-filter", "archiving-date"],
            )
            .add_step(
                "FINAL",
                "passthrough",
                deps=[
                    # "comune-extraction",
                    "intervention-start-extraction",
                    # "date-chunk-filter",
                    "archiving-date",
                ],
            )
            .make_dag()
        )


def get_pipeline():
    """Return the main pipeline as a Directed Acyclic Graph."""
    llm_model_id = "google/gemma-3-27b-it"
    llm_provider = "vllm"
    llm_model_temp = 0.05
    with sklearn.config_context(transform_output="pandas"):
        return Pipeline(
            [
                (
                    "vllm",
                    VLLM_Preprocessing(
                        vlm_provider="vllm",
                        vlm_model_id="ibm-granite/granite-vision-3.3-2b",
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
                        "comune",
                        {
                            "INDIRIZZO",
                            "CODICE_POSTALE",
                            "LUOGO",
                        },
                        load_comune,
                    ),
                ),
                ("merge", ChunksToText()),
                (
                    "extract-comune",
                    ComuneExtractor(
                        llm_provider, llm_model_id, llm_model_temp
                    ),
                ),
            ],
        )
