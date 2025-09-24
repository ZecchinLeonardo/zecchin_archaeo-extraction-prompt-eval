"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

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


class Protocollo(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sulla data e il numero di protocollo. È possibile trovare questo tipo di informazioni nel testo."""

    NrProt: int
    DataProt: str
    
class IdentificaProtocollo(dspy.Signature):
    """Identifica il numero e la data di protocollo.
    
    Cerca una stringa come 'Protocollo ... del ...' o 'Prot.  ... del ...'

    La data è nel formato gg/mm/aaaa.

    Cerca all'inizio o alla fine del testo
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    NumeroProtocollo: str = dspy.OutputField(desc="Il numero di protocollo.")
    DataProtocollo: str = dspy.OutputField(desc="La data di protocollo.")

class ProtocolloInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the date of protocollo.

    Find in the text a string like "Protocollo n. ... del ..." or "Prot. n. ... del ..."
    """

    fragmenti_relazione: str


class ProtocolloOutputData(pydantic.BaseModel):
    """A predicted number and date of protocollo."""

    NumeroProtocollo: str
    DataProtocollo: str
    


class FindProtocollo(dspy.Module):
    """DSPy model for the extraction of  Protocollo."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_protocollo = dspy.ChainOfThought(IdentificaProtocollo)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_protocollo(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )

        NR_UNIDENTIFIED = "%CHECK REQUIRED%"
        DATE_UNIDENTIFIED = "%CHECK REQUIRED%"

        numero = cast(str, predicted_output.get("NumeroProtocollo", NR_UNIDENTIFIED))
        data = cast(str, predicted_output.get("DataProtocollo", DATE_UNIDENTIFIED))

        # Return the prediction
        return to_prediction(
            ProtocolloOutputData(
                NumeroProtocollo=numero,
                DataProtocollo=data,
            )
        )


class ProtocolloExtractor(
    FieldExtractor[
        ProtocolloInputData,
        ProtocolloOutputData,
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
            ProtocolloInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
                            Prot. 9 Pisa 7 n. 1546 del 12/05/1998.""",
            ),
            ProtocolloOutputData(
                NumeroProtocollo="9 Pisa 7 n. 15467",
                DataProtocollo="12/05/1998"
            ),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindProtocollo(),
            example,
            ProtocolloOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "building__Protocollo", "building__Data_Protocollo"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        score = 0.7 * int(predicted.NumeroProtocollo == expected.NumeroProtocollo) + 0.3 * int(predicted.DataProtocollo == expected.DataProtocollo) 
        return score, TRESHOLD

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected ProtocolloOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        numero= dspy_output.get("numero_protocollo") or dspy_output.get("pred_numero_protocollo") or ""
        data = dspy_output.get("data_protocollo") or dspy_output.get("pred_data_protocollo") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want


        return ProtocolloOutputData(
            numero_protocollo=numero,
            data_protocollo=data,
            method=method  # Only include this if your schema expects it!
        )

    @override
    def _to_dspy_input(self, x) -> ProtocolloInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return ProtocolloInputData(fragmenti_relazione="", protocollo_numero_raw=None, protocollo_data_raw=None)
        row = full_row.iloc[0]
        return ProtocolloInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            protocollo_numero_raw=getattr(row, "building__Protocollo", None),
            protocollo_data_raw=getattr(row, "building__Data_Protocollo", None),
        )
    
##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["building__Protocollo"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, ProtocolloOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.building__Protocollo is not None:  # Skip if no ground truth
                protocollo_nr = t.building__Protocollo

            if t.building__Data_Protocollo is not None:
                protocollo_data = t.building__Data_Protocollo  # Skip if no ground truth

                # print(f"Nome: {nome_base}, Cognome: {cognome_base}, Iniziale: {iniziale_base}")  # Print nome, cognome, and iniziale
                
                result[InterventionId(t.id)] = ProtocolloOutputData(
                    NumeroProtocollo=protocollo_nr,
                    DataProtocollo=protocollo_data,
                )
        return result
    
##############################################################################