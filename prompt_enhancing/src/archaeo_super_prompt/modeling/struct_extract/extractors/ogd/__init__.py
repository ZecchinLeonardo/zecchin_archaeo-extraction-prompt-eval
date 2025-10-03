"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
# from archaeo_super_prompt.dataset.thesauri import load_comune_with_provincie
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


class OGD(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sulla data e il numero di protocollo. È possibile trovare questo tipo di informazioni nel testo."""

    description: str
    
class IdentificaOGD(dspy.Signature):
    """Identifica se sono presenti tracce archeologiche nella relazione.

    Cerca una stringa come "privo di" o "senza rinvenimenti" o "senza ritrovamenti" o "non sono stati rinvenuti" o "non sono stati trovati" o "nulla è stato rinvenuto" o "nulla è stato trovato" o "niente è stato rinvenuto" o "niente è stato trovato" o "nessun reperto è stato rinvenuto" o "nessun reperto è stato trovato" o "assenza di reperti" o simile

    Se non la trovi cerca una stringa come "sito pluristratificato"

    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    descrizione: str = dspy.OutputField(desc="Descrizione della natura dei rinvenimenti")

class OGDInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about what was found during the survey.

    Find in the text a string like "privo di" or "senza rinvenimenti" or "senza ritrovamenti" or "non sono stati rinvenuti" or "non sono stati trovati" or "nulla è stato rinvenuto" or "nulla è stato trovato" or "niente è stato rinvenuto" or "niente è stato trovato" or "nessun reperto è stato rinvenuto" or "nessun reperto è stato trovato" or "assenza di reperti" or similar
    """

    fragmenti_relazione: str

class OGDOutputData(pydantic.BaseModel):
    """A predicted description of what was found during the survey."""

    OGD: str  

class FindOGD(dspy.Module):
    """DSPy model for the extraction of  Protocollo."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_OGD = dspy.ChainOfThought(IdentificaOGD)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_OGD(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )

        # Define the strings to match
        negative_patterns = [
            "privo di", "senza rinvenimenti", "senza ritrovamenti",
            "non sono stati rinvenuti", "non sono stati trovati",
            "nulla è stato rinvenuto", "nulla è stato trovato",
            "niente è stato rinvenuto", "niente è stato trovato",
            "nessun reperto è stato rinvenuto", "nessun reperto è stato trovato",
            "assenza di reperti"
        ]

        # Check if the input matches any of the negative patterns
        if any(pattern in fragmenti_relazione.lower() for pattern in negative_patterns):
            descrizione = "area priva di tracce archeologiche"
        else:
            descrizione = "sito pluristratificato"

        # Return the prediction
        return to_prediction(
            OGDOutputData(
                OGD=descrizione
            )
        )

class OGDExtractor(
    FieldExtractor[
        OGDInputData,
        OGDOutputData,
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
            OGDInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
                            Nel sito non sono stati rinvenuti reperti archeologici.""",
            ),
            OGDOutputData(
                OGD="area priva di tracce archeologiche"
            ),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindOGD(),
            example,
            OGDOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__OGD"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95

        if expected.OGD==predicted.OGD and expected.OGD == "area priva di tracce archeologiche":
            score = int(predicted.OGD == expected.OGD)
        elif (expected.OGD == "sito pluristratificato" and expected.OGD == predicted.OGD) or \
            (expected.OGD != "area priva di tracce archeologiche" and predicted.OGD != "area priva di tracce archeologiche"):
            score = 1.0
        else:
            score = 0.0

        return score, TRESHOLD

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected ProtocolloOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        OGD_desc= dspy_output.get("OGD") or dspy_output.get("pred_OGD") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want


        return OGDOutputData(
            OGD=OGD_desc,
            method=method  # Only include this if your schema expects it!
        )

    @override
    def _to_dspy_input(self, x) -> OGDInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return OGDInputData(fragmenti_relazione="", OGD_descr_raw=None)
        row = full_row.iloc[0]
        return OGDInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            OGD_descr_raw=getattr(row, "university__OGD", None),
        )
    
##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__OGD"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, OGDOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.university__OGD is not None:  # Skip if no ground truth
                ogd_desc = t.university__OGD

                # print(f"Nome: {nome_base}, Cognome: {cognome_base}, Iniziale: {iniziale_base}")  # Print nome, cognome, and iniziale

                result[InterventionId(t.id)] = OGDOutputData(
                    OGD=ogd_desc,
                )
    
        return result
    
##############################################################################