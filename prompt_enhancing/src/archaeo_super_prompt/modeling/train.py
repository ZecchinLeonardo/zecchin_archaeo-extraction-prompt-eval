"""DAGs to train the FieldExtractor models."""

from typing import NamedTuple, cast

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.field_extractor import (
    FieldExtractor,
)
from .struct_extract.legacy_extractor.main_transformer import MagohDataExtractor
from .struct_extract import language_model as lm_provider_mod

from ..dataset.thesauri import load_comune, load_esecutore_candidates, load_protocollo_candidates, load_tipo_candidates
from ..types.pdfpaths import PDFPathDataset
from ..utils.result import get_model_store_dir
from .DAG_builder import DAGBuilder, DAGComponent
from .entity_extractor import NerModel, NeSelector
from .pdf_to_text import VLLM_Preprocessing
from .struct_extract.chunks_to_text import ChunksToText
from .struct_extract.extractors.archiving_date import ArchivingDateProvider
from .struct_extract.extractors.comune import ComuneExtractor
from .struct_extract.extractors.intervention_date import (
    InterventionStartExtractor,
)
from .struct_extract.extractors.esecuzione import EsecutoreExtractor
from .struct_extract.extractors.protocollo import ProtocolloExtractor
from .struct_extract.extractors.tipo import TipoExtractor

class ExtractionDAGParts(NamedTuple):
    """A decomposition of the general DAG into different parts for a better handling between the training, the inference and the evaluation modes."""
    preprocessing_root: DAGBuilder
    extraction_parts: list[tuple[DAGComponent[FieldExtractor], DAGComponent]]
    final_component: tuple[DAGComponent, list[DAGComponent]]


def get_training_dag(include_legacy: bool = False) -> ExtractionDAGParts:
    """Return the most advanced pre-processing DAG for the model.

    All its estimators and transformers are initialized with particular
    parametres.

    Return:
        A part of the complete DAG for getting the pre-processed data.
        The field extractors related to their parent node, to apply on these extractors special training or evaluation operations or to bind them to the preprocessing dag
        The final union component to finish the building of the complete DAG
        in inference mode.
    """
    llm_model_id = "google/gemma-3-27b-it"
    llm_provider = "vllm"
    llm_model_temp = 0.05

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
        InterventionStartExtractor(llm_provider, llm_model_id, llm_model_temp),
    )   
    comune_extractor = DAGComponent(
        "comune-Extractor",
        ComuneExtractor(llm_provider, llm_model_id, llm_model_temp),
    )
    comune_chunk_filter = DAGComponent(
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
    )
    comune_chunk_merger = DAGComponent("comune-CM", ChunksToText())

    intervention_date_entrypoint = DAGComponent(
        "interv-start-entrypoint", "passthrough"
    )

    esecutore_chunk_filter = DAGComponent(
        "esecutore-CF",
        NeSelector(
            "eseguito",  # <-- the NER label for your field
            {
                "ESEGUITO",
                "NOME",
                "COGNOME",
                "ORGANIZZAZIONE",
            },
            load_esecutore_candidates,  # <-- pass a function or None if not needed, e.g. for thesauri
            True,  # or False, depending on your selector logic
        ),
    )

    esecutore_chunk_merger = DAGComponent("esecutore-CM", ChunksToText())

    esecutore_extractor = DAGComponent(
        "esecutore-Extractor",
        EsecutoreExtractor(llm_provider, llm_model_id, llm_model_temp),
    )

    protocollo_chunk_filter = DAGComponent(
        "protocollo-CF",
        NeSelector(
            "protocollo",  # <-- the NER label for your field
            {
                "PROTOCOLLO",
                "NUMERO_PROTOCOLLO",
                "NR_PROT",
                "PROT_N",
            },
            load_protocollo_candidates,  # <-- pass a function or None if not needed, e.g. for thesauri
            True,  # or False, depending on your selector logic
        ),
    )

    protocollo_chunk_merger = DAGComponent("protocollo-CM", ChunksToText())

    protocollo_extractor = DAGComponent(
        "protocollo-Extractor",
        ProtocolloExtractor(llm_provider, llm_model_id, llm_model_temp),
    )

    tipo_chunk_filter = DAGComponent(
        "tipo-CF",
        NeSelector(
            "tipo",  # <-- the NER label for your field
            {
                "TIPO",
                "TIPO_DOCUMENTO",
                "TIPO_INTERVENTO",
            },
            load_tipo_candidates,  # <-- pass a function or None if not needed, e.g. for thesauri
            True,  # or False, depending on your selector logic
        ),
    )

    tipo_chunk_merger = DAGComponent("tipo-CM", ChunksToText())

    tipo_extractor = DAGComponent(
        "tipo-Extractor",
        TipoExtractor(llm_provider, llm_model_id, llm_model_temp),
    )

    final_results = DAGComponent[FieldExtractor]("FINAL", "passthrough")

    preprocessing_part = (
        DAGBuilder()
        .add_node(vllm)
        .add_node(ner, [vllm])
        .add_node(ner_featured, [vllm, ner])
        .add_node(archiving_date, [vllm])
        .add_linearly_chained_nodes(
            [comune_chunk_filter, comune_chunk_merger],
            [ner_featured],
        )
        .add_linearly_chained_nodes(
            [esecutore_chunk_filter, esecutore_chunk_merger],
            [ner_featured],
        )
        .add_linearly_chained_nodes(
            [protocollo_chunk_filter, protocollo_chunk_merger],
            [ner_featured],
        )
        .add_linearly_chained_nodes(
            [tipo_chunk_filter, tipo_chunk_merger],
            [ner_featured],
        )
        .add_linearly_chained_nodes(
            [intervention_date_chunk_filter, intervention_date_chunk_merger],
            [ner_featured],
        )
        .add_node(
            intervention_date_entrypoint,
            [intervention_date_chunk_merger, archiving_date],
        )
    )
    extraction_part = cast(
        list[tuple[DAGComponent[FieldExtractor], DAGComponent]],
        [
            # (intervention_date_extractor, intervention_date_entrypoint),
            (comune_extractor, comune_chunk_merger),
            (esecutore_extractor, esecutore_chunk_merger),
            (protocollo_extractor, protocollo_chunk_merger),
            (tipo_extractor, tipo_chunk_merger),
        ],
    )

    final_dependencies: list[DAGComponent] = [
        # archiving_date,
        # intervention_date_extractor,
        comune_extractor,
        esecutore_extractor,
        protocollo_extractor,
        tipo_extractor,
    ]

    if include_legacy:
        lm_getter = {
            "vllm": lm_provider_mod.get_vllm_model,
            "ollama": lm_provider_mod.get_ollama_model,
            "openai": lm_provider_mod.get_openai_model,
        }[llm_provider]
        legacy_extractor = DAGComponent(
            "legacy-Extractor",
            MagohDataExtractor(lm_getter(llm_model_id, llm_model_temp)),
        )
        extraction_part.append((legacy_extractor, vllm))
        final_dependencies.append(legacy_extractor)

    final_part = (final_results, final_dependencies)

    return ExtractionDAGParts(preprocessing_part, extraction_part, final_part)


def train_from_scratch(
    training_input: PDFPathDataset,
    ds: MagohDataset,
    include_legacy: bool = False,
) -> ExtractionDAGParts:
    """Return the most advanced DAG model, fitted from the data.

    Apply a training for each FieldExtractor model.
    """
    preprocessing_part, extraction_part, final_part = get_training_dag(
        include_legacy=include_legacy
    )
    preprocess_pipeline = preprocessing_part.make_dag()
    preprocessed_inputs = preprocess_pipeline.fit_transform(training_input, ds)
    
    print("preprocessed inputs:", preprocessed_inputs.keys())

    for fe_component, dep in extraction_part:
        field_extractor = fe_component.component
        if isinstance(field_extractor, str):
            # impossible
            continue
        field_extractor.fit(preprocessed_inputs[dep.component_id], ds)
        field_extractor.prompt_model_.save(
            get_model_store_dir() / f"{fe_component.component_id}.json"
        )
    return ExtractionDAGParts(preprocessing_part, extraction_part, final_part)


def get_fitted_model(
    training_input: PDFPathDataset,
    ds: MagohDataset,
    include_legacy: bool = False,
):
    """Return the most advanced DAG model, mockly fitted from the data.

    The FieldExtractor model are supposed already fitted from saved dspy
    models in get_model_store_dir() path.
    """
    preprocessing_part, extraction_part, final_part = get_training_dag(
        include_legacy=include_legacy
    )
    preprocess_pipeline = preprocessing_part.make_dag()
    preprocessed_inputs = preprocess_pipeline.fit_transform(training_input, ds)
    for fe_component, dep in extraction_part:
        field_extractor = fe_component.component
        if isinstance(field_extractor, str):
            # impossible
            continue
        field_extractor.fit(
            preprocessed_inputs[dep.component_id],
            ds,
            compiled_dspy_model_path=get_model_store_dir()
            / f"{fe_component.component_id}.json",
        )
    return ExtractionDAGParts(preprocessing_part, extraction_part, final_part)


# TODO: set the inference from the paths and the evaluation
