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

from archaeo_super_prompt.modeling.struct_extract.types import (
    BaseInputForExtraction,
    BaseInputForExtractionRowSchema,
)

import difflib
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from ...field_extractor import FieldExtractor, LLMProvider, to_prediction


# class Esecuzione(pydantic.BaseModel):
#     """Questo elemento fornisce informazioni sulla persona che ha eseguito il lavoro. È possibile trovare questo tipo di informazioni nel testo."""

#     esecutore: str
    


class Riassunto(dspy.Signature):
    """Fai il riassunto del documento.
    Fornisci solo il riassunto, senza ulteriori spiegazioni.
    Inserisci nel riassunto informazioni importanti come nomi, date, luoghi, persone coinvolte e i ritrovamenti.
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    # possibili_esecutori: list[Esecuzione] = dspy.InputField(
    #     desc="Scegliete una di queste persone che hanno eseguito i lavori archeologici."
    # )

    riassunto: str = dspy.OutputField(desc="Il riassunto del documento.")


class RiassuntoInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention which have to be summarized.
    Summary should include names, dates, places, people involved, findings.
    """

    fragmenti_relazione: str
    # possibili_esecutori: list[Esecuzione]


class RiassuntoOutputData(pydantic.BaseModel):
    """A predicted person who performed the intervention."""

    riassunto: str


class WriteSummary(dspy.Module):
    """DSPy model for the extraction of the Riassunto."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_riassunto = dspy.ChainOfThought(Riassunto)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_riassunto(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )

        UNIDENTIFIED = "%CANNOT SUMMARIZE THE DOCUMENT%"
        # Extract summary
        summary = cast(str, predicted_output.get("riassunto", UNIDENTIFIED))

        # Return the prediction
        return to_prediction(
            RiassuntoOutputData(
                riassunto=summary,
            )
        )
class InputForRiassunto(BaseInputForExtraction):
    """Riassunto input data schema."""

    summarizeIn: str


class InputForRiassuntoRowSchema(BaseInputForExtractionRowSchema):
    """When indentifying the date of an intervention, we refer first to the date of protocol."""

    summarizeInRow: str

class RiassuntoExtractor(
    FieldExtractor[
        RiassuntoInputData,
        RiassuntoOutputData,
        InputForRiassunto,
        InputForRiassuntoRowSchema,
        None,
    ]
):


# class EsecutoreExtractor(
#     FieldExtractor[
#         EsecutoreInputData,
#         EsecutoreOutputData,
#         InputForExtractionWithSuggestedThesauri,
#         InputForExtractionWithSuggestedThesauriRowSchema,
#         None,
#         # ComuneFeatSchema,
#     ]
# ):
    """Dspy-LLM-based extractor of the comune data."""

    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
    ) -> None:
        """Initialize the extractor with providing it the llm which will be used."""
        example = (
            RiassuntoInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
                                        CASTELFRANCO DI SOTTO (PI)

                                        OPEN FIBER

                                        Realizzazione, posa in opera e servizio di manutenzione di impianti in fibra ottica

                                        RELAZIONE SCIENTIFICA SUI RISULTATI DELL'ATTIVITÀ DI SORVEGLIANZA
                                        ARCHEOLOGICA

                                        1. INTRODUZIONE

                                        Le operazioni di scavo per la posa in opera della fibra ottica Open Fiber nelle località di Orentano e di Villa Campanile nel Comune di Castelfranco di Sotto (PI) sono iniziate in data 11/05/2021 e si sono concluse il giorno 24/08/2021.

                                        Le attività di scavo sono state oggetto di sorveglianza archeologica in base alle prescrizioni della Soprintendenza Archeologia, Belle Arti e Paesaggio competente per territorio e commissionate, a partire dal giorno 11 maggio, all'impresa Archeorete s.r.l.s., in sostituzione della precedente impresa affidataria dell'incarico di sorveglianza.

                                        Le attività di movimento terra eseguite hanno compreso: minitrincee, trincee con scavo tradizionale e scavo di alloggiamenti per pozzetti BUL (Banda Ultra Larga) di quattro dimensioni standard (55x55x50 cm; 76x40x76 cm; 90x70x90 cm; 125x90x107 cm).
                                        Nella parte di documentazione inerente la sorveglianza archeologica, è stata mantenuta la nomenclatura dei pozzetti adoperata da Open fiber, per cui il posizionamento degli interventi è individuabile nella pianta definitiva di progetto. Le minitrincee e le trincee, non nominate nella tavola, sono state identificate con il nome del pozzetto dal quale iniziano. Di tutti gli scavi sono state indicate le dimensioni di lunghezza, larghezza e profondità, oltre alla descrizione della stratigrafia rinvenuta.

                                        L'attività di sorveglianza archeologica ha dato esito positivo ad Orentano, in via martiri della libertà 129 dove, in occasione dello scavo per un pozzetto, è stato rinvenuto un acciottolato pertinente ad una strada antica, che è diventata oggetto di una breve indagine archeologica e stratigrafica.

                                        In tutti gli altri scavi sottoposti a sorveglianza, l'esito è stato invece negativo, non essendo state rinvenute strutture o strati di interesse archeologico; pertanto la descrizione delle stratigrafie si riferisce a strati archeologicamente sterili, dove i soli interventi antropici rinvenuti consistono in scassi per la posa in opera di sottoservizi e sono databili d epoca contemporanea.
                                    """
        ),
            RiassuntoOutputData(riassunto="Durante i lavori per la posa della fibra ottica Open Fiber a Orentano e Villa Campanile (Castelfranco di Sotto, PI), svolti tra l’11 maggio e il 24 agosto 2021 sotto sorveglianza archeologica di Archeorete s.r.l.s., è stato rinvenuto solo un acciottolato di una strada antica in via Martiri della Libertà 129 a Orentano. Tutti gli altri scavi hanno restituito stratigrafie sterili e nessun elemento di interesse archeologico.",
                                
                                )
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            WriteSummary(),
            example,
            RiassuntoOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__Eseguito_da"
        
    @override
    @override
    @staticmethod
    def field_to_be_extracted():
        # We only produce a summary; no ground-truth comparison will be performed.
        return "riassunto"

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Map the DSPy output to the RiassuntoOutputData schema.
        Accept common keys like 'riassunto' or 'summary' and fall back to an empty string.
        """
        summary = (
            dspy_output.get("riassunto")
            or dspy_output.get("summary")
            or dspy_output.get("text")
            or ""
        )

        return RiassuntoOutputData(
            riassunto=summary,
        )

    @override
    def _to_dspy_input(self, x) -> RiassuntoInputData:
        """Build the dspy input for a single intervention id dict `x`.

        Expects x to be like {'id': <intervention id>}. It will read the dataset's
        `merged_chunks` column (if present) and pass it as `fragmenti_relazione`.
        """
        intervention_id = x.get("id")
        if intervention_id is None:
            return RiassuntoInputData(fragmenti_relazione="")

        full_row = self.dataset.intervention_data[
            self.dataset.intervention_data["id"] == intervention_id
        ]
        if full_row.empty:
            return RiassuntoInputData(fragmenti_relazione="")

        row = full_row.iloc[0]
        return RiassuntoInputData(fragmenti_relazione=getattr(row, "merged_chunks", ""))