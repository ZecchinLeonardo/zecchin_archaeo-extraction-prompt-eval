"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

import difflib
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

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


class Luogo(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sul luogo. È possibile trovare questo tipo di informazioni nel testo."""

    luogo: str
    
class IdentificaLuogo(dspy.Signature):
    """Identifica il luogo dello scavo o della ricognizione.

    Cerca una stringa come "indirizzo", "località" e "ubicazione"

    Nell'indirizzo è solitamente presente una stringa come "via", "viale", "piazza", "corso", "strada", "contrada", "frazione".

    La località è solitamente vicino ad un nome di comune.

    L'ubicazione è indicata da stringhe come "presso", "nei pressi di", "vicino a", o a nomi stringhe come "poggio", "valle", "colle", "fabbrica"

    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    indirizzo: str = dspy.OutputField(desc="L'indirizzo dello scavo o della ricognizione.")
    localita: str = dspy.OutputField(desc="La località dello scavo o della ricognizione.")
    ubicazione: str = dspy.OutputField(desc="L'ubicazione dello scavo o della ricognizione.")

class LuogoInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the location and address of the survey.

    Find in the text a string like "indirizzo", "località" and "ubicazione"
    
    For address look for strings like "via", "viale", "piazza", "corso", "strada", "contrada", "frazione".
    
    For location look near names of comune.

    For location look for strings like "presso", "nei pressi di", "vicino a", or names like "poggio", "valle", "colle", "fabbrica"
    """

    fragmenti_relazione: str

class LuogoOutputData(pydantic.BaseModel):
    """A predicted description of what was found during the survey."""

    indirizzo: str  
    localita: str
    ubicazione: str

class FindLuogo(dspy.Module):
    """DSPy model for the extraction of  Protocollo."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_luogo = dspy.ChainOfThought(IdentificaLuogo)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_luogo(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )
        
        INDIRIZZO_UNIDENTIFIED = "%CHECK REQUIRED%"
        LOCALITA_UNIDENTIFIED = "%CHECK REQUIRED%"
        UBICAZIONE_UNIDENTIFIED = "%CHECK REQUIRED%"

        ind = cast(str, predicted_output.get("indirizzo", INDIRIZZO_UNIDENTIFIED))
        loc = cast(str, predicted_output.get("localita", LOCALITA_UNIDENTIFIED))
        ubi = cast(str, predicted_output.get("ubicazione", UBICAZIONE_UNIDENTIFIED))

        # Return the prediction
        return to_prediction(
            LuogoOutputData(
                indirizzo=ind,
                localita=loc,
                ubicazione=ubi
            )
        )

class LuogoExtractor(
    FieldExtractor[
        LuogoInputData,
        LuogoOutputData,
        InputForExtractionWithSuggestedThesauri,
        InputForExtractionWithSuggestedThesauriRowSchema,
        None,
        # ComuneFeatSchema,
    ]
):
    """Dspy-LLM-based extractor of the comune data."""

    # _model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
    ) -> None:
        """Initialize the extractor with providing it the llm which will be used."""
        example = (
            LuogoInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
                            Nel comune di Pisa, in via Aurelia Nord 23, in località Barbaricina, presso il fosso delle Corti, è stato rinvenuto un insediamento protostorico.""",
            ),
            LuogoOutputData(
                indirizzo="via Aurelia Nord 23",
                localita="Barbaricina",
                ubicazione="fosso delle Corti"
            ),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindLuogo(),
            example,
            LuogoOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__Ubicazione"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95

        # # Compute similarity ratio for each field (between 0 and 1)
        
        # # with DIFFLIB
        # indirizzo_sim = difflib.SequenceMatcher(None, str(predicted.indirizzo), str(expected.indirizzo)).ratio()
        # localita_sim = difflib.SequenceMatcher(None, str(predicted.localita), str(expected.localita)).ratio()
        # ubicazione_sim = difflib.SequenceMatcher(None, str(predicted.ubicazione), str(expected.ubicazione)).ratio()

        # # with rapidfuzz - token_sort_ratio
        # indirizzo_sim = fuzz.token_sort_ratio(str(predicted.indirizzo), str(expected.indirizzo)) /100
        # localita_sim = fuzz.token_sort_ratio(str(predicted.localita), str(expected.localita)) / 100
        # ubicazione_sim = fuzz.token_sort_ratio(str(predicted.ubicazione), str(expected.ubicazione)) / 100

        # # with rapidfuzz - token_set_ratio
        indirizzo_sim = fuzz.token_set_ratio(str(predicted.indirizzo), str(expected.indirizzo)) /100
        localita_sim = fuzz.token_set_ratio(str(predicted.localita), str(expected.localita)) / 100
        ubicazione_sim = fuzz.token_set_ratio(str(predicted.ubicazione), str(expected.ubicazione)) / 100

        # # with sentence-transformers
        # indirizzo_sim = cls._similarity(str(predicted.indirizzo), str(expected.indirizzo))
        # localita_sim  = cls._similarity(str(predicted.localita), str(expected.localita))
        # ubicazione_sim = cls._similarity(str(predicted.ubicazione), str(expected.ubicazione))


        # Weighted average as before
        score = 0.34 * ubicazione_sim + 0.33 * localita_sim + 0.33 * indirizzo_sim

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
        indirizzo_desc = dspy_output.get("indirizzo") or dspy_output.get("pred_indirizzo") or ""
        localita_desc = dspy_output.get("localita") or dspy_output.get("pred_localita") or ""
        ubicazione_desc = dspy_output.get("ubicazione") or dspy_output.get("pred_ubicazione") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want


        return LuogoOutputData(
            indirizzo=indirizzo_desc,
            localita=localita_desc,
            ubicazione=ubicazione_desc,
            method=method  # Only include this if your schema expects it!
        )

    @override
    def _to_dspy_input(self, x) -> LuogoInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return LuogoInputData(fragmenti_relazione="", OGD_descr_raw=None)
        row = full_row.iloc[0]
        return LuogoInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            indirizzo_descr_raw=getattr(row, "university__Indirizzo", None),
            localita_descr_raw=getattr(row, "university__Localita", None),
            ubicazione_descr_raw=getattr(row, "university__Ubicazione", None),
        )
    
##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Ubicazione"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, LuogoOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.university__Ubicazione is not None:  # Skip if no ground truth
                ubicazione_desc = t.university__Ubicazione
                indirizzo_desc = t.university__Indirizzo if t.university__Indirizzo is not None else ""
                localita_desc = t.university__Località if t.university__Località is not None else ""

                # print(f"Nome: {nome_base}, Cognome: {cognome_base}, Iniziale: {iniziale_base}")  # Print nome, cognome, and iniziale

                result[InterventionId(t.id)] = LuogoOutputData(
                    indirizzo=indirizzo_desc,
                    localita=localita_desc,
                    ubicazione=ubicazione_desc,
                )
    
        return result
    
##############################################################################
    @classmethod
    def _similarity(cls, a: str, b: str) -> float:
        """Compute semantic cosine similarity between two strings."""
        if not a or not b:
            return 0.0
        
        emb1 = cls._model.encode(a, convert_to_tensor=True)
        emb2 = cls._model.encode(b, convert_to_tensor=True)
        
        return util.cos_sim(emb1, emb2).item()