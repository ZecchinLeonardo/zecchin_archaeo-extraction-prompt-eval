"""Summary (riassunto) extractor with chunk+summarize strategy."""

import re
from typing import Any, cast, Iterable, List, override

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
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)

import difflib
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from ...field_extractor import FieldExtractor, LLMProvider, to_prediction

# new import for chunking helpers
from .chunking import chunk_text_by_tokens, estimate_token_count


class Riassunto(dspy.Signature):
    """Fai il riassunto del documento.
    Fornisci solo il riassunto, senza ulteriori spiegazioni.
    Inserisci nel riassunto informazioni importanti come nomi, date, luoghi, persone coinvolte e i ritrovamenti.
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    riassunto: str = dspy.OutputField(desc="Il riassunto del documento.")


class RiassuntoInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention which have to be summarized.
    Summary should include names, dates, places, people involved, findings.
    """

    fragmenti_relazione: str


class RiassuntoOutputData(pydantic.BaseModel):
    """A summary of the document including names, dates, places, people involved, findings."""

    riassunto: str


class WriteSummary(dspy.Module):
    """DSPy module to produce a summary using chunk+summarize strategy."""

    def __init__(self, max_chunk_tokens: int = 16000, final_max_tokens: int = 2048):
        """
        max_chunk_tokens: maximum tokens for each chunk summary job (keep well below model max).
        final_max_tokens: requested max tokens for the final summarization call (controls length).
        """
        self._estrattore_riassunto = dspy.ChainOfThought(Riassunto)
        self._max_chunk_tokens = int(max_chunk_tokens)
        self._final_max_tokens = int(final_max_tokens)

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """If the input is short, call the chain once. Otherwise chunk, summarize chunks, then summarize summaries."""
        predicted_summary = "%CANNOT_SUMMARIZE_THE_DOCUMENT%"

        # Estimate token count and decide whether chunking is needed
        try:
            n_tokens = estimate_token_count(fragmenti_relazione)
        except Exception:
            # fallback conservative estimate
            n_tokens = max(1, len(fragmenti_relazione.split()))

        # If short enough, single call
        if n_tokens <= max(1, int(self._max_chunk_tokens)):
            out = cast(dspy.Prediction, self._estrattore_riassunto(fragmenti_relazione=fragmenti_relazione))
            predicted_summary = cast(str, out.get("riassunto", predicted_summary))
            return to_prediction(RiassuntoOutputData(riassunto=predicted_summary))

        # Otherwise: chunk + summarize each chunk
        chunks: List[str] = list(chunk_text_by_tokens(fragmenti_relazione, max_tokens=self._max_chunk_tokens))
        chunk_summaries: List[str] = []
        for chunk in chunks:
            try:
                out = cast(dspy.Prediction, self._estrattore_riassunto(fragmenti_relazione=chunk))
                s = str(out.get("riassunto", "") or "")
                if s:
                    chunk_summaries.append(s)
            except Exception:
                # continue on errors in a chunk to keep the pipeline robust
                continue

        # If we got no chunk summaries, fallback to empty
        if not chunk_summaries:
            return to_prediction(RiassuntoOutputData(riassunto=""))

        # Merge chunk summaries; if merged is still too long, we may re-chunk it automatically
        merged = "\n\n".join(chunk_summaries)

        # If merged is small enough, make a final single summarization call
        try:
            merged_tokens = estimate_token_count(merged)
        except Exception:
            merged_tokens = len(merged.split())

        if merged_tokens <= max(1, int(self._final_max_tokens)):
            final_out = cast(dspy.Prediction, self._estrattore_riassunto(fragmenti_relazione=merged))
            predicted_summary = str(final_out.get("riassunto", predicted_summary))
            return to_prediction(RiassuntoOutputData(riassunto=predicted_summary))

        # Otherwise, recursively summarize merged in smaller pieces (rare)
        recursive_chunks = list(chunk_text_by_tokens(merged, max_tokens=self._final_max_tokens))
        recursive_summaries: List[str] = []
        for rc in recursive_chunks:
            try:
                out = cast(dspy.Prediction, self._estrattore_riassunto(fragmenti_relazione=rc))
                recursive_summaries.append(str(out.get("riassunto", "") or ""))
            except Exception:
                continue
        final_merged = "\n\n".join(recursive_summaries)
        if final_merged:
            final_out = cast(dspy.Prediction, self._estrattore_riassunto(fragmenti_relazione=final_merged))
            predicted_summary = str(final_out.get("riassunto", predicted_summary))
        return to_prediction(RiassuntoOutputData(riassunto=predicted_summary))


class InputForRiassunto(InputForExtractionWithSuggestedThesauri):
    """Riassunto input data schema (uses the suggested-thesauri base so the pipeline provides identified_thesaurus)."""
    # No extra fields needed: merged_chunks and identified_thesaurus come from the base


class InputForRiassuntoRowSchema(InputForExtractionWithSuggestedThesauriRowSchema):
    """Row schema (keeps the same semantics as the other extractors)."""
    # No extra fields required


class RiassuntoExtractor(
    FieldExtractor[
        RiassuntoInputData,
        RiassuntoOutputData,
        InputForRiassunto,
        InputForRiassuntoRowSchema,
        None,
    ]
):
    """Dspy-LLM-based extractor that produces a free-form summary (mapped to university__Descrizione)."""
    _model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:4")

    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
    ) -> None:
        """Initialize the extractor with the LLM info."""
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
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            WriteSummary(),  # WriteSummary handles chunking internally
            example,
            RiassuntoOutputData,
        )

    @staticmethod
    def field_to_be_extracted() -> str:
        return "university__Descrizione"

    @override
    def _transform_dspy_output(self, dspy_output: Any) -> RiassuntoOutputData:
        """
        Map the DSPy output to the RiassuntoOutputData schema.
        Accept common keys like 'riassunto' or 'summary' and fall back to an empty string.
        """
        summary = (
            dspy_output.get("descrizione")
            or dspy_output.get("pred_descrizione")
            or dspy_output.get("riassunto")
            or dspy_output.get("pred_riassunto")
            or ""
        )

        return RiassuntoOutputData(
            riassunto=summary,
        )

    @override
    def _to_dspy_input(self, x: Any) -> RiassuntoInputData:
        """
        Build the dspy input for a single preprocessed-row `x`.
        Accept dicts, pandas namedtuples/Series.
        """
        merged = ""
        intervention_id = None

        # dict-like
        if isinstance(x, dict):
            intervention_id = x.get("id", None)
            merged = x.get("merged_chunks", "") or ""
        else:
            # namedtuple / Series-like
            intervention_id = getattr(x, "id", None) or getattr(x, "Index", None)
            merged = getattr(x, "merged_chunks", None)
            if merged is None:
                try:
                    merged = x.get("merged_chunks", "")
                except Exception:
                    merged = ""

        # fallback: dataset lookup
        if (merged is None or merged == "") and intervention_id is not None:
            try:
                full_row = self.dataset.intervention_data[self.dataset.intervention_data["id"] == intervention_id]
                if not full_row.empty:
                    row = full_row.iloc[0]
                    merged = getattr(row, "merged_chunks", "") or ""
            except Exception:
                merged = merged or ""

        return RiassuntoInputData(fragmenti_relazione=merged)

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        
        # # Compute similarity ratio for each field (between 0 and 1)
        
        # # with DIFFLIB
        # riassunto_sim = difflib.SequenceMatcher(None, str(predicted.riassunto), str(expected.riassunto)).ratio()
       
        # # with rapidfuzz - token_sort_ratio
        # riassunto_sim = fuzz.token_sort_ratio(str(predicted.riassunto), str(expected.riassunto)) /100
        
        # # with rapidfuzz - token_set_ratio
        # riassunto_sim = fuzz.token_set_ratio(str(predicted.riassunto), str(expected.riassunto)) /100

        # # with sentence-transformers
        riassunto_sim = cls._similarity(str(predicted.riassunto), str(expected.riassunto))
       
        # Weighted average as before
        score = float(riassunto_sim)        
        
        # score=1.0
        score = max(0.0, min(1.0, score))
        return score, TRESHOLD

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Descrizione"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, RiassuntoOutputData]:
        result = {}
        for t in y.get_answers(ids):
            if t.university__Descrizione is not None:  # Skip if no ground truth
                descrizione = t.university__Descrizione
                result[InterventionId(t.id)] = RiassuntoOutputData(
                    riassunto=descrizione,
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