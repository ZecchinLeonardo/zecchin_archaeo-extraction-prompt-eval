"""Code containing the global model and a way to infer into it."""

import sklearn
from pandera.typing.pandas import DataFrame
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline

from ..dataset.thesauri import load_comune
from .DAG_builder import DAGComponent, DAGBuilder
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
        vllm = DAGComponent(
            "vision-lm-Reader",
            VLLM_Preprocessing(
                vlm_provider="vllm",
                vlm_model_id="ibm-granite/granite-vision-3.3-2b",
                incipit_only=True,
                prompt="OCR this part of Italian document for markdown-based processing.",
                embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
            ),
        )
        ner = DAGComponent("NER-Extractor", NerModel())
        ner_featured = DAGComponent("ner-featured", "passthrough")
        archiving_date = DAGComponent(
            "archiving-date-Oracle", ArchivingDateProvider()
        )
        intervention_date_chunk_filter = DAGComponent(
            "interv-start-CF",
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
        )
        intervention_date_chunk_merger = DAGComponent(
            "interv-start-CM", ChunksToText()
        )
        intervention_date_extractor = DAGComponent(
            "interv-start-Extractor",
            InterventionStartExtractor(
                llm_provider, llm_model_id, llm_model_temp
            ),
        )
        comune_extractor = DAGComponent(
            "comune-Extractor",
            ComuneExtractor(llm_provider, llm_model_id, llm_model_temp),
        )

        return (
            DAGBuilder()
            .add_node(vllm)
            .add_node(ner, [vllm])
            .add_node(ner_featured, [vllm, ner])
            .add_node(archiving_date, [vllm])
            .add_linearly_chained_nodes(
                [
                    DAGComponent(
                        "comune-CF",
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
                    DAGComponent("comune-CM", ChunksToText()),
                    comune_extractor,
                ],
                [ner_featured],
            )
            .add_linearly_chained_nodes(
                [
                    intervention_date_chunk_filter,
                    intervention_date_chunk_merger,
                ],
                [ner_featured],
            )
            .add_node(
                intervention_date_extractor,
                [intervention_date_chunk_merger, archiving_date],
            )
            .add_node(
                DAGComponent("FINAL", "passthrough"),
                [
                    archiving_date,
                    comune_extractor,
                    intervention_date_extractor,
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
