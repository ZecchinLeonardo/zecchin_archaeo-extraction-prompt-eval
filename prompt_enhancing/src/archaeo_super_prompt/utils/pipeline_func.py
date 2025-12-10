import os
import importlib
import shutil
import pathlib
import urllib.parse
# import uuid
# import mlflow
import pandas as pd
import ast
import re
import datetime
import calendar
from pathlib import Path
# import traceback

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from archaeo_super_prompt import visualization as visualizator
from archaeo_super_prompt.visualization import mlflow_logging as mmlflow

from archaeo_super_prompt.config.env import getenv_or_throw

from archaeo_super_prompt.dataset import MagohDataset, SamplingParams

from archaeo_super_prompt.utils.cache import get_cache_dir_for
from archaeo_super_prompt.utils.result import get_model_store_dir

import archaeo_super_prompt.modeling.train as training
import archaeo_super_prompt.modeling.predict as infering

from archaeo_super_prompt.modeling import pdf_to_text

from archaeo_super_prompt.modeling.struct_extract import language_model as lm_provider_mod
from archaeo_super_prompt.modeling.struct_extract import field_extractor as fe

# INTERVENTION_DATE Libraries from extractors
from archaeo_super_prompt.modeling.struct_extract.extractors.archiving_date import ArchivingDateProvider, ArchivingDateOutputSchema
import archaeo_super_prompt.modeling.struct_extract.extractors.intervention_date as ide
from archaeo_super_prompt.modeling.struct_extract.extractors.intervention_date import InterventionStartExtractor, ITALIAN_MONTHS, Data, DataInterventoInputData

# COMUNE Libraries from extractors
from archaeo_super_prompt.modeling.struct_extract.extractors.comune import ComuneExtractor, ComuneInputData, Comune
from archaeo_super_prompt.dataset.thesauri import comune_province as cp

# EXECUTOR Libraries from extractors
from archaeo_super_prompt.modeling.struct_extract.extractors.esecuzione import EsecutoreExtractor, EsecutoreInputData, Esecuzione

from archaeo_super_prompt.modeling.struct_extract.extractors.protocollo import ProtocolloExtractor, ProtocolloInputData, Protocollo

from archaeo_super_prompt.modeling.struct_extract.extractors.tipo import TipoExtractor, TipoInputData, Tipo

from archaeo_super_prompt.modeling.struct_extract.extractors.ogd import OGDExtractor, OGDInputData, OGD

from archaeo_super_prompt.modeling.struct_extract.extractors.luogo import LuogoExtractor, LuogoInputData, Luogo

from archaeo_super_prompt.modeling.struct_extract.extractors.year import YearExtractor, DataInterventoInputData  # AnnoInterventoInputData #, Luogo

from archaeo_super_prompt.modeling.struct_extract.extractors.direzione_funzionario import DirezioneFunzionarioExtractor, DirezioneFunzionarioInputData

# RITROVAMENTI Libraries from extractors
from archaeo_super_prompt.modeling.struct_extract.extractors.ritrovamenti import RitrovamentoExtractor, RitrovamentiInputData, Ritrovamenti
# from archaeo_super_prompt.dataset.thesauri import ritrovamenti  as rtr
from archaeo_super_prompt.dataset.thesauri import load_ritrovamento



def _as_list(v):
    if isinstance(v, list):
        return v
    if pd.isna(v):
        return []
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "[{(":
            try:
                x = ast.literal_eval(s)
                return x if isinstance(x, list) else [x]
            except Exception:
                return [v]
        return [v]
    return [v]


def _as_str_list(v):
    return [str(x) for x in _as_list(v)]


def _as_int_list(v):
    out = []
    for x in _as_list(v):
        try:
            out.append(int(x))
        except Exception:
            try:
                out.append(int(float(x)))
            except Exception:
                pass
    return out


# fixes for invalid model outputs that broke the pipeline
def _as_int(x, d):
    try:
        return int(x)
    except Exception:
        return d


def _predict_safe(self, X):
    def parse(dp):
        if dp is None:
            return datetime.date(1900, 1, 1)
        if isinstance(dp, (datetime.date, datetime.datetime, pd.Timestamp)):
            return dp.date() if hasattr(dp, "date") else dp
        s = str(dp).strip()
        if s == "" or s.lower() in ("none", "nan", "nat"):
            return datetime.date(1900, 1, 1)
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d"):
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except Exception:
                pass
        parts = re.split(r"[-/\\. ]+", s)
        try:
            if len(parts) >= 3:
                d, m, y = map(int, parts[:3])
                if y < 100:
                    y += 2000
                return datetime.date(y, m, d)
        except Exception:
            pass
        m = re.search(r"(\\d{4})", s)
        if m:
            return datetime.date(int(m.group(1)), 1, 1)
        return datetime.date(1900, 1, 1)

    rows = [{"id": a.id, "data_protocollo": parse(getattr(a, "building__Data_Protocollo", None))}
            for a in self._mds.get_answers(set(X["id"].to_list()))]
    return ArchivingDateOutputSchema.validate(pd.DataFrame(rows).set_index("id"))


# remove bad ocr output
def _strip_tables_and_noise(s):
    s = re.sub(r"^\\s*\\d+\\s*,\\s*\\d+\\s*=\\s*.*$", "", s, flags=re.M | re.S)
    s = re.sub(r"`-+`\\s*", "", s)
    return s


# avoids going over max context window for long (and badly read) documents
def _focus_and_truncate(s, date_re, max_chars=20000):
    s = _strip_tables_and_noise(s)
    if len(s) <= max_chars:
        return s
    lines = s.splitlines()
    hits = [ln for ln in lines if date_re.search(ln)]
    head = "\n".join(lines[:4000])
    tail = "\n".join(lines[-2000:])
    middle = "\n".join(hits)[:8000]
    out = "\n".join([head, middle, tail])
    return out[:max_chars]


# COMUNE functions

# comune fixing section

def load_comune_aligned():
    df = pd.read_csv(cp._get_comune_file())
    df = df[df["nome"].notnull() & df["provincia"].notnull()]
    return [(id_com, nome) for _, id_com, nome in df[["id_com","nome"]].itertuples()]


def _read_comuni_tables_strict():
    comuni = pd.read_csv(cp._get_comune_file())[
        ["id_com", "nome", "provincia"]
    ].rename(columns={"id_com":"comune_id","nome":"name","provincia":"province_id"})
    comuni = comuni.dropna(subset=["name","province_id"]).copy()
    comuni["comune_id"] = comuni["comune_id"].astype(int)
    comuni["province_id"] = comuni["province_id"].astype(int)

    # drop duplicated comuni keeping the first occurrence
    dups = comuni["comune_id"].duplicated(keep="first").sum()
    if dups:
        print(f"[warn] dropping {dups} duplicated comuni by comune_id")
        comuni = comuni.drop_duplicates(subset=["comune_id"], keep="first")

    # Ensure the province id column from the CSV (`id_prov`) is mapped to `province_id`.
    # Previously this incorrectly attempted to rename `province_id` -> `province_id`,
    # which left `province_id` missing and triggered a KeyError.
    province = pd.read_csv(cp._get_provincie_file())[
        ["id_prov","nome","sigla"]
    ].rename(columns={"id_prov":"province_id","nome":"province_name"})
    province["province_id"] = province["province_id"].astype(int)
    province = province.drop_duplicates(subset=["province_id"], keep="first")

    # each comune refers to exactly one province
    merged = comuni.merge(
        province[["province_id","province_name","sigla"]],
        on="province_id",
        how="left",
        validate="m:1",   # raise if a province_id maps to multiple province rows
        copy=False,
    )

    comuni = comuni.set_index("comune_id").sort_index()
    merged = merged.set_index("comune_id").sort_index()
    return comuni, merged


def _mk_from_pos(merged, pos2id, fields, pos: int):
    if not (0 <= pos < len(pos2id)):
        return None
    cid = pos2id[pos]
    if cid not in merged.index:
        return None
    # scalar gets:
    name = str(merged.at[cid, "name"])
    prov_name = str(merged.at[cid, "province_name"])
    sigla = str(merged.at[cid, "sigla"])
    kw = {}
    if "citta_nome" in fields:
        kw["citta_nome"] = name
    if "provicia_nome" in fields:
        kw["provicia_nome"] = prov_name
    if "provincia_sigla" in fields:
        kw["provincia_sigla"] = sigla
    if "id" in fields:
        kw["id"] = int(cid)
    # not sure if name or nome is used
    if "nome" in fields and "citta_nome" not in fields:
        kw["nome"] = name
    if "name" in fields and "citta_nome" not in fields:
        kw["name"] = name
    return Comune(**kw)


def _comune_to_dspy_input(self, x):
    pos_list = getattr(x, "identified_thesaurus", None) or getattr(x, "possibili_comuni", None) or []
    if not isinstance(pos_list, list):
        pos_list = [pos_list]
    pos_list = [int(v) for v in pos_list if str(v).isdigit()]
    cands = [c for c in (_mk_from_pos(p) for p in pos_list) if c is not None]
    return ComuneInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
        possibili_comuni=cands,
    )


# INTERVENTION_DATE functions

def _clamp_month(m):
    m = _as_int(m, 1)
    return 1 if m < 1 else 12 if m > 12 else m

def _clamp_day(y, m, d):
    y = _as_int(y, 1900)
    m = _clamp_month(m)
    d = _as_int(d, 1)
    last = calendar.monthrange(max(1, y), m)[1]
    return 1 if d < 1 else last if d > last else d

def _get_min_date_safe(o):
    y = _as_int(getattr(o, "start_year", 1900), 1900)
    m = _clamp_month(getattr(o, "start_month", 1))
    d = _clamp_day(y, m, getattr(o, "start_day", 1))
    return datetime.date(y, m, d)

def _get_max_date_safe(o):
    y = getattr(o, "end_year", None)
    if y is None:
        y = getattr(o, "start_year", 1900)
    y = _as_int(y, 1900)
    m = _clamp_month(getattr(o, "end_month", 12))
    d = _clamp_day(y, m, getattr(o, "end_day", 28))
    return datetime.date(y, m, d)


# feed trimmed "fragmenti relazione" and fix month (went out of bounds)
def _to_dspy_input_patched(self, x):
    d = x.data_protocollo
    return DataInterventoInputData(
        fragmenti_relazione=_focus_and_truncate(x.merged_chunks),
        data_di_archiviazone=Data(
            giorno=int(getattr(d, "day", d.day)),
            mese=ITALIAN_MONTHS[int(getattr(d, "month", d.month)) - 1],
            anno=int(getattr(d, "year", d.year)),
        ),
    )

# load scans CSV with a robust loader (if you don't already have one)
def load_scans_safe(path):
    # df = pd.read_csv(path, encoding='utf-8-sig')
    # # drop leading saved index column if it looks numeric
    # if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
    #     sample = df.iloc[:,0].dropna().astype(str).head(20).tolist()
    #     if sample and all(s.strip().lstrip("-").isdigit() for s in sample):
    #         df = df.iloc[:,1:].copy()
    # df.columns = df.columns.str.strip()
    # if "id" in df.columns:
    #     df["id"] = pd.to_numeric(df["id"].astype(str).str.strip(), errors="coerce").astype("Int64")
    # return df
    df = pd.read_csv(path, encoding='utf-8-sig')
    # drop leading saved index column if it looks numeric
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        sample = df.iloc[:,0].dropna().astype(str).head(20).tolist()
        if sample and all(s.strip().lstrip("-").isdigit() for s in sample):
            df = df.iloc[:,1:].copy()
    df.columns = df.columns.str.strip()
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"].astype(str).str.strip(), errors="coerce").astype("Int64")

    # sanitize text/embedding columns so they are not NaN
    for col in ("chunk_embedding_content", "chunk_content", "filename"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # also ensure chunk_type / chunk_page_position columns are sane
    if "chunk_type" in df.columns:
        df["chunk_type"] = df["chunk_type"].apply(_as_str_list)
    if "chunk_page_position" in df.columns:
        df["chunk_page_position"] = df["chunk_page_position"].apply(_as_int_list)

    return df


def _esecutore_to_dspy_input(self, x):
    return EsecutoreInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
        # possibili_esecutori=_as_list(getattr(x, "possibili_esecutori", [])),
    )

# EsecutoreExtractor._to_dspy_input = _esecutore_to_dspy_input


def _protocollo_to_dspy_input(self, x):
    return ProtocolloInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

# ProtocolloExtractor._to_dspy_input = _protocollo_to_dspy_input


def _tipo_to_dspy_input(self, x):
    return TipoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

# TipoExtractor._to_dspy_input = _tipo_to_dspy_input


def _ogd_to_dspy_input(self, x):
    return TipoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

# OGDExtractor._to_dspy_input = _ogd_to_dspy_input


def _luogo_to_dspy_input(self, x):
    return LuogoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

# LuogoExtractor._to_dspy_input = _luogo_to_dspy_input


def _year_to_dspy_input(self, x):
    return DataInterventoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

# YearExtractor._to_dspy_input = _year_to_dspy_input


def _direzione_funzionario_to_dspy_input(self, x):
    return DirezioneFunzionarioInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )
# DirezioneFunzionarioExtractor._to_dspy_input = _direzione_funzionario_to_dspy_input

# RITROVAMENTI functions

def _ritrovamenti_to_dspy_input(self, x):
    # Gather candidate positions (may come from identified_thesaurus or
    # possibly_ritrovamenti depending on pipeline wiring).
    pos_list = getattr(x, "identified_thesaurus", None) or getattr(
        x, "possibili_ritrovamenti", None
    ) or []
    if not isinstance(pos_list, list):
        pos_list = [pos_list]

    # Normalize to integer positions where possible, ignore non-numeric entries
    normalized_positions: list[int] = []
    for v in pos_list:
        try:
            normalized_positions.append(int(v))
        except Exception:
            # skip values that are not convertible to int
            continue

    # Try to find a thesaurus attached to the extractor instance; fallback to
    # loading the ritrovamenti thesaurus if not present.
    thes = getattr(self, "_thesaurus", None)
    if thes is None:
        try:
            

            thes = load_ritrovamento()
        except Exception:
            thes = []

    # Build candidate Ritrovamenti objects by indexing into the thesaurus.
    candidates: list = []
    if hasattr(thes, "iloc"):
        # pandas DataFrame/Series-like
        for p in normalized_positions:
            if 0 <= p < len(thes):
                row = thes.iloc[p]
                # prefer possible column names for the label
                label = None
                for col in ("iii_livello", "iii_lev", "label", "name"):
                    try:
                        if col in row.index:
                            label = row[col]
                            break
                    except Exception:
                        # row may be a scalar/Series without index
                        pass
                if label is None:
                    # try first element
                    try:
                        label = row.iloc[0]
                    except Exception:
                        label = str(row)
                candidates.append(Ritrovamenti(iii_livello=str(label)))
    else:
        # assume list/iterable of tuples or strings
        for p in normalized_positions:
            try:
                item = thes[p]
            except Exception:
                continue
            label = None
            # if tuple-like, take first element as label
            if isinstance(item, (list, tuple)) and len(item) > 0:
                label = item[0]
            elif isinstance(item, dict):
                # try common keys
                for key in ("iii_livello", "iii_lev", "label", "name"):
                    if key in item:
                        label = item[key]
                        break
            else:
                label = item
            candidates.append(Ritrovamenti(iii_livello=str(label)))

    return RitrovamentiInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
        possibili_ritrovamenti=candidates,
    )


# RitrovamentoExtractor._to_dspy_input = _ritrovamenti_to_dspy_input