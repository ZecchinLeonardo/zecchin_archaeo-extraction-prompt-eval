"""Comune LLM extractor."""

import re
from typing import cast, override, List

import dspy
import pydantic
from pandera.typing.pandas import Series


from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.dataset.thesauri import load_ritrovamento, load_cronologia
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
    crono: str = ""
    


class IdentificaRitrovamento(dspy.Signature):
    """
    Identifica i ritrovamenti rinvenuti durante lo scavo descritto in questi frammenti di relazione. 
    I ritrovamenti possibili sono identificati da stringhe tipo 'è stato rinvenuto', 'è stato ritrovato', 'i ritrovamenti', 'ci sono tracce di'.
    Identifica anche le cronologie associate ai ritrovamenti.
    Le cronologie possono essere espresse in modi diversi, come 'età del bronzo', 'periodo romano', 'neolitico', ecc. o con numeri interi che indicano anni o secoli.
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    possibili_ritrovamenti: list[Ritrovamenti] = dspy.InputField(
        desc="Scegliete un di questi ritrovamenti"
    )
    ritrovamento: list[str] = dspy.OutputField(desc="Lista dei ritrovamenti")
    cronologia: list[str] = dspy.OutputField(desc="Lista delle cronologie dei ritrovamenti")
    


class RitrovamentiInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about what was found during the excavation.
    The findings are identified by strings like 'è stato rinvenuto', 'è stato ritrovato', 'i ritrovamenti', 'ci sono tracce di'.
    Identify also the chronologies associated with the findings.
    The chronologies can be expressed in different ways, such as 'età del bronzo', 'periodo romano', 'neolitico', etc. or with integers indicating years or centuries
    """

    fragmenti_relazione: str
    possibili_ritrovamenti: list[Ritrovamenti]


class RitrovamentiOutputData(pydantic.BaseModel):
    """A predicted ritrovamento where the intervention took place."""
    finding: List[str] = pydantic.Field(default_factory=list)
    chronology: List[str] = pydantic.Field(default_factory=list)
   


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
        UNIDENTIFIED_CHRONOLOGY = "%CHECK_REQUIRED%"

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

        raw_findings = predicted_output.get("ritrovamenti", predicted_output.get("ritrovamento", predicted_output.get("finding", UNIDENTIFIED_FINDING)))
        raw_chronologies = predicted_output.get("cronologie", predicted_output.get("cronologia", predicted_output.get("chronology", UNIDENTIFIED_CHRONOLOGY)))

        # mapped_finding = _map_to_thesaurus_label(raw_finding)
        # return to_prediction(
        #     RitrovamentiOutputData(
        #         finding=cast(str, mapped_finding),
        #         chronology=cast(str, raw_chronology),

        #     )
        # )

        # normalize into lists
        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i) for i in x if i is not None and str(i).strip() != ""]
            s = str(x)
            # split common separators if returned as single string
            parts = re.split(r"\s*[,;|\n]\s*", s)
            return [p for p in (p.strip() for p in parts) if p]

        findings_list = _to_list(raw_findings)
        chrono_list = _to_list(raw_chronologies)

        # align lengths: if chronologies shorter, pad with empty strings
        if len(chrono_list) < len(findings_list):
            chrono_list += [""] * (len(findings_list) - len(chrono_list))
        if len(findings_list) < len(chrono_list):
            findings_list += [""] * (len(chrono_list) - len(findings_list))

        mapped_findings = [_map_to_thesaurus_label(f) for f in findings_list]

        def _map_chronology(text: str) -> str:
            """Normalize chronology and map to `thesaurus_crono.csv` when possible."""
            if not text:
                return ""
            raw = str(text)
            n = _norm(raw)
            try:
                cr = load_cronologia()
                # pick a reasonable label column if present
                col_candidates = [
                    "name_full",
                    "name",
                    "cronologia",
                    "label",
                ]
                col = next((c for c in col_candidates if c in cr.columns), cr.columns[0])
                cr = cr.assign(_norm=cr[col].fillna("").apply(_norm))
                mask = cr["_norm"] == n
                if mask.any():
                    return str(cr.loc[mask, col].iloc[0])
                if _rf_process is not None:
                    choices = cr["_norm"].tolist()
                    best = _rf_process.extractOne(n, choices, score_cutoff=80)
                    if best:
                        matched_norm = best[0]
                        return str(cr[cr["_norm"] == matched_norm][col].iloc[0])
            except Exception:
                pass
            return raw

        mapped_chronos = [_map_chronology(c) for c in chrono_list]

        return to_prediction(
            RitrovamentiOutputData(
                finding=[cast(str, m) for m in mapped_findings],
                chronology=[cast(str, c) for c in mapped_chronos],
            )
        )


# -- SKlearn part


class RitrovamentoFeatSchema(BasePerInterventionFeatureSchema):
    """Extracted data about the Ritrovamento."""

    ritrovamento_id: int
    chrono_feat: str


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
È stato rinvenuto un insediamento dell'età del bronzo con tracce materiali e tracce di sepoltura dell'età medievale.""",
                possibili_ritrovamenti=[
                    Ritrovamenti(
                        iii_livello="insediamento",
                    )
                ],
            ),
            RitrovamentiOutputData(
                finding=["insediamento", "ad inumazione"],
                chronology=["età del bronzo", "età medievale"],
                ),
        )
        # TODO: load this more lazily
        self._thesaurus = load_ritrovamento()
        # print("columns:", self._thesaurus.columns.tolist())
        # print(self._thesaurus.head().to_string(index=False))
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
        # label_candidates = ["iii_livello", "iii_lev", "university__iii_lev"]
        label_candidates = [
            "III Livello",
            "III livello",
            "III_Livello",
            "III_livello",
            "iii_livello",
            "iii_lev",
            "university__iii_lev",
        ]
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
        # label_candidates = ["iii_lev", "iii_livello", "university__iii_lev"]
        # code_candidates = [
        #     "iii_lev_code",
        #     "iii_livello_id",
        #     "university__iii_lev_code",
        #     "iii_livello_code",
        # ]
        label_candidates = [
            "III Livello",
            "III livello",
            "III_Livello",
            "III_livello",
            "iii_lev",
            "iii_livello",
            "university__iii_lev",
        ]
        code_candidates = [
            "iii_lev_code",
            "iii_livello_id",
            "university__iii_lev_code",
            "iii_livello_code",
            # tolerate uppercase/spacing variants just in case
            "III_Livello_code",
            "III Livello_code",
            "III_Livello_id",
            "III Livello id",
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
        if "finding" not in df.columns and "ritrovamenti" not in df.columns and "ritrovamento" not in df.columns:
            raise ValueError("dspy output does not contain 'finding' / 'ritrovamento(s)' field")

        # # Merge predicted finding string with thesaurus to get the id
        # merged = (
        #     df.reset_index()
        #     .merge(th_df[["iii_lev", "ritrovamento_id"]], left_on="finding", right_on="iii_lev", how="left")
        #     [["id", "ritrovamento_id"]]
        #     .set_index("id")
        # )

        # # fill missing mappings with -1 and ensure integer dtype
        # merged["ritrovamento_id"] = merged["ritrovamento_id"].fillna(-1).astype(int)

        # return RitrovamentoFeatSchema.validate(merged)

        # normalize columns to 'finding' and 'chronology'
        if "ritrovamenti" in df.columns and "finding" not in df.columns:
            df = df.rename(columns={"ritrovamenti": "finding"})
        if "ritrovamento" in df.columns and "finding" not in df.columns:
            df = df.rename(columns={"ritrovamento": "finding"})
        if "cronologie" in df.columns and "chronology" not in df.columns:
            df = df.rename(columns={"cronologie": "chronology"})
        if "cronologia" in df.columns and "chronology" not in df.columns:
            df = df.rename(columns={"cronologia": "chronology"})
        if "chronology" not in df.columns:
            df["chronology"] = [[] for _ in range(len(df))]

        # helper to coerce cell into list
        def _to_list_cell(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i) for i in x if i is not None and str(i).strip() != ""]
            s = str(x)
            return [p for p in re.split(r"\s*[,;|\n]\s*", s) if p.strip()]

        # build pairs and explode so that findings and chronologies stay aligned
        df["pairs"] = df.apply(lambda r: list(zip(_to_list_cell(r.get("finding", [])), _to_list_cell(r.get("chronology", [])))), axis=1)
        df = df.explode("pairs").reset_index()
        # when no pairs, explode yields NaN
        df[["finding", "chronology"]] = pd.DataFrame(df["pairs"].tolist(), index=df.index).fillna("")

        # Merge predicted finding string with thesaurus to get the id
        merged = (
            df.merge(th_df[["iii_lev", "ritrovamento_id"]], left_on="finding", right_on="iii_lev", how="left")
            .rename(columns={"chronology": "chrono_feat"})
            .loc[:, ["id", "ritrovamento_id", "chrono_feat"]]
        )

        # fill missing mappings with -1 and ensure integer dtype
        merged["ritrovamento_id"] = merged["ritrovamento_id"].fillna(-1).astype(int)
        merged["chrono_feat"] = merged["chrono_feat"].fillna("").astype(str)

        merged = merged.set_index("id")

        return RitrovamentoFeatSchema.validate(merged)

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        try:
            pred_set = set(predicted.finding)
            exp_set = set(expected.finding)
            match = int(pred_set == exp_set)
        except Exception:
            match = 0
        return match, TRESHOLD
        # return int(predicted.finding == expected.finding), TRESHOLD

    # @override
    # @classmethod
    # def filter_training_dataset(
    #     cls, y: MagohDataset, ids: set[InterventionId]
    # ) -> set[InterventionId]:
    #     return y.filter_good_records_for_training(
    #         ids,
    #         lambda df: cast(Series[bool], df["III Livello"].notnull()),
    #     )
    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        """
        Return the subset of `ids` that have at least one non-empty III Livello
        in the findings table (y.findings). This ensures the trainset is not empty
        when the ground-truth lives in the findings table.
        """
        label_candidates = [
            "III Livello",
            "III livello",
            "III_Livello",
            "III_livello",
            "iii_livello",
            "iii_lev",
        ]

        findings = getattr(y, "findings", None)
        if findings is None or findings.empty:
            return set()

        # find which column to use
        found_col = next((c for c in label_candidates if c in findings.columns), None)
        if found_col is None:
            return set()

        # select findings rows with non-null labels and collect intervention ids
        valid_rows = findings[findings[found_col].notnull()]
        if "scheda_intervento.id" not in valid_rows.columns:
            return set()

        valid_ids = set()
        for v in valid_rows["scheda_intervento.id"].unique():
            try:
                if pd.notna(v):
                    valid_ids.add(int(v))
            except Exception:
                continue

        # intersect with provided ids (accept InterventionId or int)
        out: set[InterventionId] = set()
        for i in ids:
            try:
                ii = int(i)
            except Exception:
                continue
            if ii in valid_ids:
                out.add(InterventionId(ii))
        return out

    # @override
    # @classmethod
    # def _select_answers(
    #     cls, y: MagohDataset, ids: set[InterventionId]
    # ) -> dict[InterventionId, RitrovamentiOutputData]:
    #     def to_ritrovamenti_data(row) -> RitrovamentiOutputData:
    #         """Retrieve the answer value and map it to the canonical thesaurus label.

    #         Returns an empty finding when none is available.
    #         """
    #         default_output = RitrovamentiOutputData(finding="")
    #         if row is None:
    #             return default_output

    #         def _norm(s: str) -> str:
    #             s = "" if s is None else str(s)
    #             s = unicodedata.normalize("NFKD", s)
    #             s = "".join(ch for ch in s if not unicodedata.combining(ch))
    #             return re.sub(r"\s+", " ", s).strip().lower()

    #         def _map_to_thesaurus_label(text: str) -> str:
    #             if not text:
    #                 return ""
    #             if isinstance(text, str) and text.strip().startswith("{"):
    #                 try:
    #                     parsed = ast.literal_eval(text)
    #                     if isinstance(parsed, dict) and "finding" in parsed:
    #                         text = parsed["finding"]
    #                 except Exception:
    #                     pass
    #             raw = str(text)
    #             n = _norm(raw)
    #             try:
    #                 th = load_ritrovamento()
    #                 th = th.assign(iii_lev_norm=th["iii_lev"].fillna("").apply(_norm))
    #                 mask = th["iii_lev_norm"] == n
    #                 if mask.any():
    #                     return str(th.loc[mask, "iii_lev"].iloc[0])
    #                 if _rf_process is not None:
    #                     choices = th["iii_lev_norm"].tolist()
    #                     best = _rf_process.extractOne(n, choices, score_cutoff=80)
    #                     if best:
    #                         matched_norm = best[0]
    #                         return str(th[th["iii_lev_norm"] == matched_norm]["iii_lev"].iloc[0])
    #             except Exception:
    #                 pass
    #             return raw

    #         # Try label fields first
    #         for c in ("university__iii_lev", "iii_livello", "iii_lev"):
    #             val = getattr(row, c, None)
    #             if val:
    #                 try:
    #                     mapped = _map_to_thesaurus_label(val)
    #                     return RitrovamentiOutputData(finding=str(mapped))
    #                 except Exception:
    #                     return default_output

    #         # Try code fields and map to label via thesaurus
    #         for code_field in ("university__iii_lev_code", "iii_lev_code", "iii_livello_id"):
    #             code_val = getattr(row, code_field, None)
    #             if code_val is not None and str(code_val).strip() != "":
    #                 try:
    #                     th = load_ritrovamento()
    #                     mask = th["iii_lev_code"].astype(str) == str(code_val)
    #                     matched = th[mask]
    #                     if not matched.empty:
    #                         label = matched.iloc[0]["iii_lev"]
    #                         mapped = _map_to_thesaurus_label(label)
    #                         return RitrovamentiOutputData(finding=str(mapped))
    #                 except Exception:
    #                     try:
    #                         return RitrovamentiOutputData(finding=str(code_val))
    #                     except Exception:
    #                         return default_output

    #         return default_output

    #     return {InterventionId(t.id): to_ritrovamenti_data(t) for t in y.get_answers(ids)}


    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, RitrovamentiOutputData]:
        def to_ritrovamenti_data_from_row(row) -> tuple[str, str]:
            """Return (label, chronology) tuple from a findings-row-like dict/obj."""
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
            
            # retrieve label from findings table: prefer "III Livello"
            label_cols = ["III Livello", "III livello", "III_Livello", "III_livello", "IIIlivello"]
            chrono_cols = ["Datazione", "Datazione Finale", "Datazione_Finale", "datazione"]

            # retrieve label
            for c in label_cols:
                try:
                    val = row.get(c) if hasattr(row, "get") else (row[c] if c in row else None)
                except Exception:
                    val = None
                if val and str(val).strip() != "":
                    mapped = _map_to_thesaurus_label(val)
                    # chronology from the same finding row (if present)
                    chrono = ""
                    for cc in chrono_cols:
                        try:
                            chrono_val = row.get(cc) if hasattr(row, "get") else (row[cc] if cc in row else None)
                        except Exception:
                            chrono_val = None
                        if chrono_val and str(chrono_val).strip() != "":
                            chrono = str(chrono_val)
                            break
                    return str(mapped), str(chrono or "")

            # fallback: try code-like id column 'ID'
            try:
                code_val = row.get("ID") if hasattr(row, "get") else (row["ID"] if "ID" in row else None)
            except Exception:
                code_val = None
            if code_val is not None and str(code_val).strip() != "":
                try:
                    th = load_ritrovamento()
                    mask = th["iii_lev_code"].astype(str) == str(code_val)
                    matched = th[mask]
                    if not matched.empty:
                        label = matched.iloc[0]["iii_lev"]
                        return _map_to_thesaurus_label(label), ""
                except Exception:
                    return str(code_val), ""

            return "", ""

        out: dict[InterventionId, RitrovamentiOutputData] = {}
        findings_df = y.findings  # raw findings table returned by get_entries / get_entries_with_ids
        for iid in ids:
            try:
                rows = findings_df[findings_df["scheda_intervento.id"] == int(iid)]
            except Exception:
                rows = pd.DataFrame()

            labels: List[str] = []
            chronos: List[str] = []
            if rows is not None and len(rows) > 0:
                for _, r in rows.iterrows():
                    label, chrono = to_ritrovamenti_data_from_row(r)
                    if label:
                        labels.append(label)
                        chronos.append(chrono)
            else:
                # fallback to intervention-level answers
                try:
                    answers = y.get_answers({iid})
                    ans = next(iter(answers.values()), None)
                    if ans is not None:
                        if getattr(ans, "finding", None):
                            labels = [str(getattr(ans, "finding"))] if not isinstance(getattr(ans, "finding"), list) else list(getattr(ans, "finding"))
                        if getattr(ans, "chronology", None):
                            chronos = [str(getattr(ans, "chronology"))] if not isinstance(getattr(ans, "chronology"), list) else list(getattr(ans, "chronology"))
                except Exception:
                    pass

            out[iid] = RitrovamentiOutputData(finding=labels, chronology=chronos)

        return out

    @override
    @staticmethod
    def field_to_be_extracted():
        return "III Livello"
