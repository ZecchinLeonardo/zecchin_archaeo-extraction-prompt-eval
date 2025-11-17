"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.dataset.thesauri import load_ritrovamento
from archaeo_super_prompt.modeling.struct_extract.types import (
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...field_extractor import FieldExtractor, LLMProvider, to_prediction
import pandas as pd
import unicodedata
import ast
try:
    from rapidfuzz import process as _rf_process
except Exception:
    _rf_process = None

# -- DSPy part


class Ritrovamenti(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sui ritrovamenti. È possibile trovare questo tipo di informazioni nel testo."""

    iii_livello: str
    


class IdentificaRitrovamento(dspy.Signature):
    """
    Identifica i ritrovamenti rinvenuti durante lo scavo descritto in questi frammenti di relazione. 
    I ritrovamenti possibili sono identificati da stringhe tipo 'è stato rinvenuto', 'è stato ritrovato', 'i ritrovamenti', 'ci sono tracce di'.
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    possibili_ritrovamenti: list[Ritrovamenti] = dspy.InputField(
        desc="Scegliete un di questi ritrovamenti"
    )
    ritrovamento: str = dspy.OutputField(desc="Il nome completo del ritrovamento")
    


class RitrovamentiInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about what was found during the excavation.
    The findings are identified by strings like 'è stato rinvenuto', 'è stato ritrovato', 'i ritrovamenti', 'ci sono tracce di'.
    """

    fragmenti_relazione: str
    possibili_ritrovamenti: list[Ritrovamenti]


class RitrovamentiOutputData(pydantic.BaseModel):
    """A predicted ritrovamento where the intervention took place."""
    finding: str
   


class FindRitrovamento(dspy.Module):
    """DSPy model for the extraction of the comune."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_di_ritrovamento = dspy.ChainOfThought(IdentificaRitrovamento)

    def forward(
        self, fragmenti_relazione: str, possibili_ritrovamenti: list[Ritrovamenti]
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_di_ritrovamento(
                fragmenti_relazione=fragmenti_relazione,
                possibili_ritrovamenti=possibili_ritrovamenti,
            ),
        )
        UNIDENTIFIED_FINDING = "%CHECK_REQUIRED%"

        def _norm(s: str) -> str:
            s = "" if s is None else str(s)
            s = unicodedata.normalize("NFKD", s)
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            return re.sub(r"\s+", " ", s).strip().lower()

        def _map_to_thesaurus_label(text: str) -> str:
            if not text:
                return ""
            # try to extract dict-like 'finding' if present
            if isinstance(text, str) and text.strip().startswith("{"):
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict) and "finding" in parsed:
                        text = parsed["finding"]
                except Exception:
                    pass

            raw = str(text)
            n = _norm(raw)
            try:
                th = load_ritrovamento()
                th = th.assign(iii_lev_norm=th["iii_lev"].fillna("").apply(_norm))
                # exact normalized match
                mask = th["iii_lev_norm"] == n
                if mask.any():
                    return str(th.loc[mask, "iii_lev"].iloc[0])
                # fuzzy match if available
                if _rf_process is not None:
                    choices = th["iii_lev_norm"].tolist()
                    best = _rf_process.extractOne(n, choices, score_cutoff=80)
                    if best:
                        matched_norm = best[0]
                        return str(th[th["iii_lev_norm"] == matched_norm]["iii_lev"].iloc[0])
            except Exception:
                pass
            # fallback: return original text
            return raw
            # return UNIDENTIFIED_FINDING

        raw = predicted_output.get("ritrovamento", predicted_output.get("finding", UNIDENTIFIED_FINDING))
        mapped = _map_to_thesaurus_label(raw)
        return to_prediction(
            RitrovamentiOutputData(
                finding=cast(str, mapped)
            )
        )


# -- SKlearn part


class RitrovamentoFeatSchema(BasePerInterventionFeatureSchema):
    """Extracted data about the Ritrovamento."""

    ritrovamento_id: int


class RitrovamentoExtractor(
    FieldExtractor[
        RitrovamentiInputData,
        RitrovamentiOutputData,
        InputForExtractionWithSuggestedThesauri,
        InputForExtractionWithSuggestedThesauriRowSchema,
        RitrovamentoFeatSchema,
    ]
):
    """Dspy-LLM-based extractor of the ritrovamento data."""

    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
    ) -> None:
        """Initialize the extractor with providing it the llm which will be used."""
        example = (
            RitrovamentiInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 7 :
È stato rinvenuto un insediamento dell'età del bronzo con tracce materiali.""",
                possibili_ritrovamenti=[
                    Ritrovamenti(
                        iii_livello="insediamento",
                    )
                ],
            ),
            RitrovamentiOutputData(finding="insediamento"),
        )
        # TODO: load this more lazily
        self._thesaurus = load_ritrovamento()
        print("columns:", self._thesaurus.columns.tolist())
        print(self._thesaurus.head().to_string(index=False))
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindRitrovamento(),
            example,
            RitrovamentiOutputData,
        )

    @override
    def _to_dspy_input(self, x) -> RitrovamentiInputData:
        ritrovamenti = self._thesaurus
        possible_ritrovamenti = ritrovamenti.iloc[x.identified_thesaurus]
        # Be tolerant to different column names in the thesaurus slice.
        label_candidates = ["iii_livello", "iii_lev", "university__iii_lev"]
        chosen_label = next((c for c in label_candidates if c in possible_ritrovamenti.columns), None)
        if chosen_label is None:
            # fallback to first column
            chosen_label = possible_ritrovamenti.columns[0]

        values = possible_ritrovamenti[chosen_label].astype(str).tolist()

        return RitrovamentiInputData(
            fragmenti_relazione=x.merged_chunks,
            possibili_ritrovamenti=[Ritrovamenti(iii_livello=cast(str, v)) for v in values],
        )
             
    @override
    def _transform_dspy_output(self, y):
        # Transform the set of dspy outputs into a DataFrame with a single
        # integer column `ritrovamento_id` validated by `RitrovamentoFeatSchema`.
        # The thesaurus (`self._thesaurus`) may be provided as a pandas DataFrame
        # with columns `iii_lev` and `iii_lev_code`, or as a list of tuples
        # (iii_lev, iii_lev_code). Be permissive and coerce to a DataFrame.
        thes = self._thesaurus
        # Coerce thesaurus to DataFrame and normalize column names to iii_lev / iii_lev_code
        if hasattr(thes, "columns"):
            th_df = thes.copy()
        else:
            try:
                th_df = pd.DataFrame(thes)
            except Exception:
                raise ValueError("unable to coerce self._thesaurus into a DataFrame")

        # possible source names
        label_candidates = ["iii_lev", "iii_livello", "university__iii_lev"]
        code_candidates = [
            "iii_lev_code",
            "iii_livello_id",
            "university__iii_lev_code",
            "iii_livello_code",
        ]

        label_col = next((c for c in label_candidates if c in th_df.columns), None)
        code_col = next((c for c in code_candidates if c in th_df.columns), None)

        if label_col is None or code_col is None:
            raise ValueError(f"thesaurus missing expected label/code columns; found {th_df.columns.tolist()}")

        th_df = th_df.rename(columns={label_col: "iii_lev", code_col: "iii_lev_code"})

        # ensure codes are numeric when possible
        try:
            th_df = th_df.assign(ritrovamento_id=th_df["iii_lev_code"].astype(int))
        except Exception:
            th_df = th_df.assign(ritrovamento_id=th_df["iii_lev_code"])

        df = self._identity_output_set_transform_to_df(y)

        # expected column from dspy output is 'finding'
        if "finding" not in df.columns:
            raise ValueError("dspy output does not contain 'finding' field")

        # Merge predicted finding string with thesaurus to get the id
        merged = (
            df.reset_index()
            .merge(th_df[["iii_lev", "ritrovamento_id"]], left_on="finding", right_on="iii_lev", how="left")
            [["id", "ritrovamento_id"]]
            .set_index("id")
        )

        # fill missing mappings with -1 and ensure integer dtype
        merged["ritrovamento_id"] = merged["ritrovamento_id"].fillna(-1).astype(int)

        return RitrovamentoFeatSchema.validate(merged)

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        return int(predicted.finding == expected.finding), TRESHOLD

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__iii_lev"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, RitrovamentiOutputData]:
        def to_ritrovamenti_data(row) -> RitrovamentiOutputData:
            """Retrieve the answer value and map it to the canonical thesaurus label.

            Returns an empty finding when none is available.
            """
            default_output = RitrovamentiOutputData(finding="")
            if row is None:
                return default_output

            def _norm(s: str) -> str:
                s = "" if s is None else str(s)
                s = unicodedata.normalize("NFKD", s)
                s = "".join(ch for ch in s if not unicodedata.combining(ch))
                return re.sub(r"\s+", " ", s).strip().lower()

            def _map_to_thesaurus_label(text: str) -> str:
                if not text:
                    return ""
                if isinstance(text, str) and text.strip().startswith("{"):
                    try:
                        parsed = ast.literal_eval(text)
                        if isinstance(parsed, dict) and "finding" in parsed:
                            text = parsed["finding"]
                    except Exception:
                        pass
                raw = str(text)
                n = _norm(raw)
                try:
                    th = load_ritrovamento()
                    th = th.assign(iii_lev_norm=th["iii_lev"].fillna("").apply(_norm))
                    mask = th["iii_lev_norm"] == n
                    if mask.any():
                        return str(th.loc[mask, "iii_lev"].iloc[0])
                    if _rf_process is not None:
                        choices = th["iii_lev_norm"].tolist()
                        best = _rf_process.extractOne(n, choices, score_cutoff=80)
                        if best:
                            matched_norm = best[0]
                            return str(th[th["iii_lev_norm"] == matched_norm]["iii_lev"].iloc[0])
                except Exception:
                    pass
                return raw

            # Try label fields first
            for c in ("university__iii_lev", "iii_livello", "iii_lev"):
                val = getattr(row, c, None)
                if val:
                    try:
                        mapped = _map_to_thesaurus_label(val)
                        return RitrovamentiOutputData(finding=str(mapped))
                    except Exception:
                        return default_output

            # Try code fields and map to label via thesaurus
            for code_field in ("university__iii_lev_code", "iii_lev_code", "iii_livello_id"):
                code_val = getattr(row, code_field, None)
                if code_val is not None and str(code_val).strip() != "":
                    try:
                        th = load_ritrovamento()
                        mask = th["iii_lev_code"].astype(str) == str(code_val)
                        matched = th[mask]
                        if not matched.empty:
                            label = matched.iloc[0]["iii_lev"]
                            mapped = _map_to_thesaurus_label(label)
                            return RitrovamentiOutputData(finding=str(mapped))
                    except Exception:
                        try:
                            return RitrovamentiOutputData(finding=str(code_val))
                        except Exception:
                            return default_output

            return default_output

        return {InterventionId(t.id): to_ritrovamenti_data(t) for t in y.get_answers(ids)}

    @override
    @staticmethod
    def field_to_be_extracted():
        return "university__iii_lev"
