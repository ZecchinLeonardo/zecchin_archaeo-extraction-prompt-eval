"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

import difflib

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.dataset.thesauri import load_comune_with_provincie
from archaeo_super_prompt.modeling.struct_extract.types import (
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId


from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...field_extractor import FieldExtractor, LLMProvider, to_prediction


# -- DSPy part


class Tipo(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sul tipo di intervento e sul tipo di documento. 

    È possibile trovare questo tipo di informazioni nel testo."""

    Documento: str
    Intervento: str
    
class IdentificaTipo(dspy.Signature):
    """Identifica il tipo di documento e il tipo di intervento.

    Cerca una stringa come 'Tipologia di documento: ...' e 'Tipologia di intervento: ...'

    La tipologia di intervento contiene parole chiave come 'scavo', 'ricognizione', assistenza'

    La tipologia di documento contiene parole chiave come 'relazione'

    Cerca all'inizio del testo
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    TipoDocumento: str = dspy.OutputField(desc="Il tipo di documento.")
    TipoIntervento: str = dspy.OutputField(desc="Il tipo di intervento.")

class TipoInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the type of intervention and type of document.

    For type of intervention look for keywords like "scavo", "ricognizione", "assistenza"
    
    For type of document look for keywords like "relazione"
    """

    fragmenti_relazione: str


class TipoOutputData(pydantic.BaseModel):
    """A predicted type of document and type of intervention."""

    TipoDocumento: str
    TipoIntervento: str


class FindTipo(dspy.Module):
    """DSPy model for the extraction of  Tipo."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_tipo = dspy.ChainOfThought(IdentificaTipo)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_tipo(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )

        DOCUMENT_UNIDENTIFIED = "%CHECK REQUIRED%"
        INTERVENTION_UNIDENTIFIED = "%CHECK REQUIRED%"

        document = cast(str, predicted_output.get("TipoDocumento", DOCUMENT_UNIDENTIFIED))
        intervention = cast(str, predicted_output.get("TipoIntervento", INTERVENTION_UNIDENTIFIED))

        # Return the prediction
        return to_prediction(
            TipoOutputData(
                TipoDocumento=document,
                TipoIntervento=intervention,
            )
        )


class TipoExtractor(
    FieldExtractor[
        TipoInputData,
        TipoOutputData,
        InputForExtractionWithSuggestedThesauri,
        InputForExtractionWithSuggestedThesauriRowSchema,
        None,
        # ComuneFeatSchema,
    ]
):
    """Dspy-LLM-based extractor of the comune data."""

    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
    ) -> None:
        """Initialize the extractor with providing it the llm which will be used."""
        example = (
            TipoInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
                            Tipologia di documento: Relazione di scavo
                            Tipologia di intervento: Scavo preventivo""",
            ),
            TipoOutputData(
                TipoDocumento="Relazione di scavo",
                TipoIntervento="Scavo preventivo"
            ),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindTipo(),
            example,
            TipoOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__Tipo_di_intervento", "building__Tipo_di_documento"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95

        # Compute similarity ratio for each field (between 0 and 1)
        tipo_intervento_sim = difflib.SequenceMatcher(None, str(predicted.TipoIntervento), str(expected.TipoIntervento)).ratio()
        tipo_documento_sim = difflib.SequenceMatcher(None, str(predicted.TipoDocumento), str(expected.TipoDocumento)).ratio()
        
        # Weighted average as before
        score = 0.5 * tipo_intervento_sim + 0.5 * tipo_documento_sim

        # Defensive clamp
        score = max(0.0, min(1.0, score))

        return score, TRESHOLD

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected ProtocolloOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        documento= dspy_output.get("tipo_documento") or dspy_output.get("pred_tipo_documento") or ""
        intervento = dspy_output.get("tipo_intervento") or dspy_output.get("pred_tipo_intervento") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want


        return TipoOutputData(
            TipoDocumento=documento,
            TipoIntervento=intervento,
            method=method  # Only include this if your schema expects it!
        )

    @override
    def _to_dspy_input(self, x) -> TipoInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return TipoInputData(fragmenti_relazione="", tipo_documento_raw=None, tipo_intervento_raw=None)
        row = full_row.iloc[0]
        return TipoInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            tipo_documento_raw=getattr(row, "university__Tipo_di_intervento", None),
            tipo_intervento_raw=getattr(row, "building__Tipo_di_documento", None),
        )
    
##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Tipo_di_intervento"].notnull()),
            # lambda df: cast(Series[bool], df["building__Tipo_di_documento"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, TipoOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.building__Tipo_di_documento is not None:  # Skip if no ground truth
                tipo_documento = t.building__Tipo_di_documento

            if t.university__Tipo_di_intervento is not None:  # Skip if no ground truth
                tipo_intervento = t.university__Tipo_di_intervento

                # print(f"Nome: {nome_base}, Cognome: {cognome_base}, Iniziale: {iniziale_base}")  # Print nome, cognome, and iniziale

                result[InterventionId(t.id)] = TipoOutputData(
                    TipoDocumento=tipo_documento,
                    TipoIntervento=tipo_intervento,
                )
        return result
    
##############################################################################