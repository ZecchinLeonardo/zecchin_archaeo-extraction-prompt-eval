"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.types import (
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)

import difflib
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from ...field_extractor import FieldExtractor, LLMProvider, to_prediction


# @override
# @staticmethod
def split_nome_cognome_initial(nome_cognome: str) -> tuple[str, str, str]:
    """
    Splits a full name string into nome and cognome, and returns initial of nome.
    """
    if not nome_cognome:
        return "", "", ""
    
    parts = nome_cognome.strip().split()
    if len(parts) == 1:
        nome = parts[0]
        cognome = ""
    else:
        nome = parts[0]
        cognome = " ".join(parts[1:])
    
    initial = nome[0] if nome else ""
    
    return nome, cognome, initial
# -- DSPy part


class Esecuzione(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sulla persona che ha eseguito il lavoro. È possibile trovare questo tipo di informazioni nel testo."""

    esecutore: str
    


class IdentificaEsecutore(dspy.Signature):
    """Identifica la persona che ha eseguito i lavori archeologici descritti in questi frammenti di relazione.
    
    Cerca una stringa come "Eseguito da" 
    
    Se manca la stringa "Eseguito da" cerca una sigla sull'intestazione della pagina.

    Può essere identificato da sigle come "s.n.c.", "s.r.l.", "S.p.A.", "S.a.s.", "Dott.", "Ing.", "Arch.", o da parole come "società", "cooperativa", "consorzio", "università", "museo".
    
    Considera solo il nome e il cognome senza titoli come "Dott.", "Ing.", "Arch."
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    # possibili_esecutori: list[Esecuzione] = dspy.InputField(
    #     desc="Scegliete una di queste persone che hanno eseguito i lavori archeologici."
    # )

    esecutore: str = dspy.OutputField(desc="Il nome completo di chi ha eseguito i lavori.")


class EsecutoreInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the person who carried out the operations.
    
    Find in the text a string like "Eseguito da"

    If the string is missing, find a mark or a stamp on the header of the page
    
    It can be identified by abbreviations such as "s.n.c.", "s.r.l.", "S.p.A.", "S.a.s.", "Dott.", "Ing.", "Arch.", or by words like "società", "cooperativa", "consorzio", "università", "museo".

    Consider only the first and last name without titles like "Dott.", "Ing.", "Arch."
    """

    fragmenti_relazione: str
    # possibili_esecutori: list[Esecuzione]


class EsecutoreOutputData(pydantic.BaseModel):
    """A predicted person who performed the intervention."""

    nome_cognome: str
    nome: str
    cognome: str
    iniziale: str


class FindEsecutore(dspy.Module):
    """DSPy model for the extraction of the Esecutore."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_esecuzione = dspy.ChainOfThought(IdentificaEsecutore)

    def forward(
        self, fragmenti_relazione: str#, possibili_esecutori: list[Esecuzione]
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_esecuzione(
                fragmenti_relazione=fragmenti_relazione,
                #possibili_esecutori=possibili_esecutori,
            ),
        )

        UNIDENTIFIED = "%CHECK REQUIRED%"
        # Extract nome_cognome and split it into components
        nome_cognome = cast(str, predicted_output.get("esecutore", UNIDENTIFIED))
        nome, cognome, iniziale = split_nome_cognome_initial(nome_cognome)

        # Return the prediction
        return to_prediction(
            EsecutoreOutputData(
                nome_cognome=nome_cognome,
                nome=nome,
                cognome=cognome,
                iniziale=iniziale,
            )
        )


class EsecutoreExtractor(
    FieldExtractor[
        EsecutoreInputData,
        EsecutoreOutputData,
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
            EsecutoreInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
L'intervento è stato eseguito dal dott. Mario Rossi in data 12/05/2023.""",
                # possibili_esecutori=[
                #     Esecuzione(
                #         esecutore="Mario Rossi",
                #         # provicia_nome="Lucca",
                #         # provincia_sigla="LU",
                #     )
                # ],
            ),
            EsecutoreOutputData(nome_cognome="Mario Rossi",
                                nome="Mario",
                                cognome="Rossi",
                                iniziale="M"),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindEsecutore(),
            example,
            EsecutoreOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__Eseguito_da"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        # score = 0.8 * int(predicted.cognome == expected.cognome) + 0.2 * int(predicted.iniziale == expected.iniziale)
        # score = fuzz.token_sort_ratio(predicted.nome_cognome, expected.nome_cognome) / 100
        score = 0.8 *(fuzz.token_sort_ratio(predicted.cognome, expected.cognome) / 100) + 0.2 * (fuzz.token_sort_ratio(predicted.iniziale, expected.iniziale) / 100)



        return score, TRESHOLD

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected EsecutoreOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        nome_cognome = dspy_output.get("esecutore") or dspy_output.get("nome_cognome") or dspy_output.get("pred_nome_cognome") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want

        nome, cognome, iniziale = split_nome_cognome_initial(nome_cognome)

        return EsecutoreOutputData(
            nome_cognome=nome_cognome,
            nome=nome,
            cognome=cognome,
            iniziale=iniziale,
            method=method  # Only include this if your schema expects it!
        )

    @override
    def _to_dspy_input(self, x) -> EsecutoreInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return EsecutoreInputData(fragmenti_relazione="", esecutore_raw=None)
        row = full_row.iloc[0]
        return EsecutoreInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            esecutore_raw=getattr(row, "university__Eseguito_da", None),
        )
    
##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Eseguito_da"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, EsecutoreOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.university__Eseguito_da is not None:  # Skip if no ground truth
                nome_cognome_base = t.university__Eseguito_da
                nome_base, cognome_base, iniziale_base = split_nome_cognome_initial(nome_cognome_base)
                # print(f"Nome: {nome_base}, Cognome: {cognome_base}, Iniziale: {iniziale_base}")  # Print nome, cognome, and iniziale
                result[InterventionId(t.id)] = EsecutoreOutputData(
                    nome_cognome=nome_cognome_base,
                    nome=nome_base,
                    cognome=cognome_base,
                    iniziale=iniziale_base,
                )
        return result
    
##############################################################################