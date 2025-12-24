"""Comune LLM extractor."""

import re
from typing import cast, override, Optional

import dspy
import pydantic
from pandera.typing.pandas import Series

from rapidfuzz import fuzz

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.types import (
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...field_extractor import FieldExtractor, LLMProvider, to_prediction

from archaeo_super_prompt.utils.chunk_extractor import (
    _choose_best_chunk_and_page
)


# @override
# @staticmethod

# -- DSPy part


# class Esecuzione(pydantic.BaseModel):
#     """Questo elemento fornisce informazioni sulla persona che ha eseguito il lavoro. È possibile trovare questo tipo di informazioni nel testo."""

#     esecutore: str
    


class IdentificaDirezioneFunzionario(dspy.Signature):
    """Identifica la persona che ha diretto l'intervento archeologico e il funzionario competente.
    
    Cerca una stringa come "Direzione scientifica", "Direttore scientifico", "Funzionario responsabile", "Funzionario competente", "Responsabile archeologo", "Responsabile scientifico". 
    
    Se mancano queste stringhe, ritorno "%CHECK REQUIRED%".

    Può essere identificato da sigle come "Dott.", "Ing.", "Arch.".
    
    Considera solo il nome e il cognome senza titoli come "Dott.", "Ing.", "Arch."
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    direzione: str = dspy.OutputField(desc="Il nome completo di chi ha diretto i lavori.")
    funzionario: str = dspy.OutputField(desc="Il nome completo del funzionario competente.")


class DirezioneFunzionarioInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the person who directed the intervention.
    Find in the text a string like "Direzione scientifica", "Direttore scientifico", "Funzionario responsabile", "Funzionario competente", "Responsabile archeologo", "Responsabile scientifico".

    If the string is missing, return "%CHECK REQUIRED%".
    
    It can be identified by abbreviations such as "Dott.", "Ing.", "Arch.".

    Consider only the first and last name without titles like "Dott.", "Ing.", "Arch."
    """

    fragmenti_relazione: str


class DirezioneFunzionarioOutputData(pydantic.BaseModel):
    """A predicted person who performed the intervention."""

    direzione: str
    funzionario: str
    chunk: str = ""
    page_number: Optional[int] = None


class FindDirezioneFunzionario(dspy.Module):
    """DSPy model for the extraction of the Direzione scientifica and Funzionario competente."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_esecuzione = dspy.ChainOfThought(IdentificaDirezioneFunzionario)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_esecuzione(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )

        UNIDENTIFIED = "%CHECK REQUIRED%"
        # Extract nome_cognome and split it into components
        direzione = cast(str, predicted_output.get("direzione", UNIDENTIFIED))
        funzionario = cast(str, predicted_output.get("funzionario", UNIDENTIFIED))

        chunk, page = _choose_best_chunk_and_page(fragmenti_relazione, direzione)


        # Return the prediction
        return to_prediction(
            DirezioneFunzionarioOutputData(
                direzione=direzione,
                funzionario=funzionario,
                chunk=chunk,
                page_number=page,
            )
        )


class DirezioneFunzionarioExtractor(
    FieldExtractor[
        DirezioneFunzionarioInputData,
        DirezioneFunzionarioOutputData,
        InputForExtractionWithSuggestedThesauri,
        InputForExtractionWithSuggestedThesauriRowSchema,
        None,
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
            DirezioneFunzionarioInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
L'intervento è stato diretto dal dott. Francesco Bianchi, il funzionario competente dell'area della Valdarno inferiore era Giovanni Verdi.""",
            ),
            DirezioneFunzionarioOutputData(
                                direzione="Francesco Bianchi",
                                funzionario="Giovanni Verdi",
                                chunk="L'intervento è stato diretto dal dott. Francesco Bianchi, il funzionario competente dell'area della Valdarno inferiore era Giovanni Verdi.",
                                page_number=1,
                                ),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindDirezioneFunzionario(),
            example,
            DirezioneFunzionarioOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "building__Funzionario_competente"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        # score = 0.7 * int(predicted.funzionario == expected.funzionario) + 0.3 * int(predicted.direzione == expected.direzione)
        score = 0.7 *(fuzz.token_sort_ratio(predicted.funzionario, expected.funzionario) / 100) + 0.3 * (fuzz.token_sort_ratio(predicted.direzione, expected.direzione) / 100)
        return score, TRESHOLD

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected EsecutoreOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        pred_direzione = dspy_output.get("direzione") or dspy_output.get("direttore") or dspy_output.get("pred_direzione") or ""
        pred_funzionario = dspy_output.get("funzionario") or dspy_output.get("pred_funzionario") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want

        chunk = (
            dspy_output.get("chunk")
            or dspy_output.get("esecutore_chunk")
            or dspy_output.get("fragment")
            or ""
        )
        page = dspy_output.get("page") or dspy_output.get("page_number") or None

        

        return DirezioneFunzionarioOutputData(
            direzione=pred_direzione,
            funzionario=pred_funzionario,
            method=method,  # Only include this if your schema expects it!
            chunk=chunk,
            page_number=page,
        )

    @override
    def _to_dspy_input(self, x) -> DirezioneFunzionarioInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return DirezioneFunzionarioInputData(fragmenti_relazione="")#, esecutore_raw=None)
        row = full_row.iloc[0]
        return DirezioneFunzionarioInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            # esecutore_raw=getattr(row, "university__Eseguito_da", None),
        )
    
##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["building__Funzionario_competente"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DirezioneFunzionarioOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.building__Funzionario_competente is not None:  # Skip if no ground truth
                funzionario_truth = t.building__Funzionario_competente
            else:
                funzionario_truth = ""
            if t.university__Direzione_scientifica is not None:
                direzione_truth = t.university__Direzione_scientifica
            else:
                direzione_truth = ""
            
            result[InterventionId(t.id)] = DirezioneFunzionarioOutputData(
                    direzione=direzione_truth,
                    funzionario=funzionario_truth,
                )
        return result
    
##############################################################################