"""
Converted from notebook: exploratory/4.1.1-complete_pipeline.ipynb
This script preserves the original notebook cell order and content.
Run it from the repository root or adjust working directory as needed.
"""

# Stuff to run before this notebook

# local pc, make sure postgresql is running with the necessary data
# ssh -i ./fair_key -R 5432:localhost:5432 -L 8889:localhost:8889 -L 8887:localhost:8887 -L 8050:localhost:8050 name@host

# for conda env and vllm venv

# conda activate /raid/ggattiglia/magoh_ai/env
# source ./vllm-server/.venv/bin/activate

# LLM, needs conda and venv, replace cuda devices with gpus and hf token with your token (need to request permission to use model on huggingface first)
# CUDA_VISIBLE_DEVICES=0,1 HF_TOKEN=[hf_token] vllm serve --port 8001 google/gemma-3-27b-it --tensor-parallel-size 2 --gpu-memory-utilization 0.6

# mlflow, needs conda
# mlflow server --host 127.0.0.1 --port 8887

# NER server, needs conda and venv
# cd prompt_enhancing/models/custom-remote-models/src/magoh_ai_sup_server
# just run-server

# the functions and code in this notebook should be in the code itself

# # IPython magics: try to enable autoreload when running interactively
# try:
#     from IPython import get_ipython
#     ip = get_ipython()
#     if ip is not None:
#         ip.run_line_magic("load_ext", "autoreload")
#         ip.run_line_magic("autoreload", "2")
# except Exception:
#     # Not running inside IPython; ignore
#     pass

import os
import importlib
import shutil
import pathlib
import urllib.parse
import uuid
import mlflow
import pandas as pd
import ast
import re
import datetime
import calendar
from pathlib import Path
import traceback

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

# change working directory similar to the notebook
os.chdir(os.path.join(os.getcwd(),  '..', 'src'))
print("Current working directory:", os.getcwd())


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
def _focus_and_truncate(s, max_chars=20000):
    s = _strip_tables_and_noise(s)
    if len(s) <= max_chars:
        return s
    lines = s.splitlines()
    hits = [ln for ln in lines if DATE_RE.search(ln)]
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


def _mk_from_pos(pos: int):
    if not (0 <= pos < len(_POS2ID)):
        return None
    cid = _POS2ID[pos]
    if cid not in _MERGED.index:
        return None
    # scalar gets:
    name = str(_MERGED.at[cid, "name"])
    prov_name = str(_MERGED.at[cid, "province_name"])
    sigla = str(_MERGED.at[cid, "sigla"])
    kw = {}
    if "citta_nome" in _FIELDS:
        kw["citta_nome"] = name
    if "provicia_nome" in _FIELDS:
        kw["provicia_nome"] = prov_name
    if "provincia_sigla" in _FIELDS:
        kw["provincia_sigla"] = sigla
    if "id" in _FIELDS:
        kw["id"] = int(cid)
    # not sure if name or nome is used
    if "nome" in _FIELDS and "citta_nome" not in _FIELDS:
        kw["nome"] = name
    if "name" in _FIELDS and "citta_nome" not in _FIELDS:
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


# class LoadScans(BaseEstimator, TransformerMixin):
#     def __init__(self, cache_csv: Path):
#         self.cache_csv = Path(cache_csv)
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         scans = pd.read_csv(self.cache_csv)
#         scans = scans.drop_duplicates(subset=["id"])
#         return X.merge(scans, on="id", how="inner")


class LoadScans(BaseEstimator, TransformerMixin):
    # def __init__(self, cache_csv: str | Path):
    #     self.cache_csv = Path(cache_csv)
    #     self._df = None
    # def fit(self, X, y=None):
    #     df = pd.read_csv(self.cache_csv)
    #     if "id" in df.columns:
    #         df["id"] = df["id"].astype("int64").astype(int)
    #     if "chunk_type" in df.columns:
    #         df["chunk_type"] = df["chunk_type"].apply(_as_str_list)
    #     if "chunk_page_position" in df.columns:
    #         df["chunk_page_position"] = df["chunk_page_position"].apply(_as_int_list)
    #     if "identified_thesaurus" in df.columns:
    #         df["identified_thesaurus"] = df["identified_thesaurus"].apply(_as_int_list)
    #     if "named_entities" in df.columns:
    #         df["named_entities"] = df["named_entities"].apply(_as_list)
    #     self._df = df
    #     return self
    # def transform(self, X):
    #     X = X.copy()
    #     if "id" in X.columns:
    #         X["id"] = X["id"].astype(int)
    #     return X.merge(self._df, on="id", how="inner")
    def __init__(self, cache_csv: str | Path):
        self.cache_csv = Path(cache_csv)
        self._df = None
    def fit(self, X, y=None):
        df = pd.read_csv(self.cache_csv)
        if "id" in df.columns:
            df["id"] = df["id"].astype("int64").astype(int)
        if "chunk_type" in df.columns:
            df["chunk_type"] = df["chunk_type"].apply(_as_str_list)
        if "chunk_page_position" in df.columns:
            df["chunk_page_position"] = df["chunk_page_position"].apply(_as_int_list)
        if "identified_thesaurus" in df.columns:
            df["identified_thesaurus"] = df["identified_thesaurus"].apply(_as_int_list)
        if "named_entities" in df.columns:
            df["named_entities"] = df["named_entities"].apply(_as_list)

        # Ensure embedding and text columns are non-null strings to satisfy schema checks
        for col in ("chunk_embedding_content", "chunk_content", "filename"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        self._df = df
        return self
    def transform(self, X):
        X = X.copy()
        if "id" in X.columns:
            X["id"] = X["id"].astype(int)
        return X.merge(self._df, on="id", how="inner")
    
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


cp.load_comune = load_comune_aligned

DATE_RE = re.compile(r"\\b(\\d{1,2}[\\/\\.-]\\d{1,2}[\\/\\.-]\\d{2,4}|gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\\b", re.I)


# cache location
csv_file = input("Enter the name of the scans CSV file (default: 'scans.csv'): ") or "scans.csv"
CACHE_CSV = get_cache_dir_for("interim", "miscel") / csv_file
SCANS_DF = pd.read_csv(CACHE_CSV)

EXP_NAME = "Complete training test"


mlflow.set_tracking_uri(f"http://{getenv_or_throw('MLFLOW_HOST')}:{getenv_or_throw('MLFLOW_PORT')}")
mlflow.set_experiment(EXP_NAME)
mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
pd.set_option('display.max_columns', None)
set_config(display="diagram")

# HARD RESET OF CACHES / ARTIFACTS run this when fresh mlflow rerun is needed
# import mlflow

# wipe compiled DSPy programs used by extractors
# from archaeo_super_prompt.utils.result import get_model_store_dir
shutil.rmtree(get_model_store_dir(), ignore_errors=True)

# wipe joblib / skdag caches inside project cache dirs
# from archaeo_super_prompt.utils.cache import get_cache_dir_for
for scope in ["external", "internal", "interim", "miscel", "thesaurus", "raw"]:
    # candidate subparts to probe; empty string requests the scope root if supported
    for subpart in ["", "pdfs", "miscel", "joblib", "__joblib_cache__", "skdag"]:
        try:
            base = get_cache_dir_for(scope, subpart)
        except Exception as e:
            # continue trying other subparts (log first failure for scope root)
            # print(f"get_cache_dir_for({scope!r}, {subpart!r}) raised: {e}")
            continue
        base = Path(base)
        if not base.exists():
            continue
        for p in base.rglob("*"):
            try:
                if p.is_dir() and any(k in str(p).lower() for k in ("joblib", "skdag", "__joblib_cache__")):
                    print("Removing cache dir:", p)
                    shutil.rmtree(p, ignore_errors=False)
            except PermissionError:
                print("Permission denied removing", p)
            except Exception as e:
                print(f"Failed removing {p}: {e}")

fresh_dir = pathlib.Path.cwd() / f"mlruns_fresh_{uuid.uuid4().hex[:6]}"
fresh_dir.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"http://{getenv_or_throw('MLFLOW_HOST')}:{getenv_or_throw('MLFLOW_PORT')}")
mlflow.set_experiment(f"fresh-{uuid.uuid4().hex[:6]}")
mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
pd.set_option('display.max_columns', None)
set_config(display="diagram")

# store_dir = get_model_store_dir()
# Path(store_dir).mkdir(parents=True, exist_ok=True)
# print("Model store:", store_dir)


store_dir = get_model_store_dir()
Path(store_dir).mkdir(parents=True, exist_ok=True)
print("Model store:", store_dir)


selected_ids = set(map(int, SCANS_DF["id"].dropna().tolist()))
ds = MagohDataset(selected_ids)
print(dir(ds))
print(ds.intervention_data.columns)
print(ds.findings.columns)

inputs = ds.files.merge(SCANS_DF[["id"]].drop_duplicates(), on="id", how="inner")
train_inputs, eval_inputs = inputs.iloc[:10], inputs.iloc[10:]
# print(f"train inputs columns are {train_inputs.columns} \n")
# print(eval_inputs.columns)

##############################################################################

scans = load_scans_safe(CACHE_CSV)
# inspect ds.files
ds.files["id"] = pd.to_numeric(ds.files["id"].astype(str).str.strip(), errors="coerce").astype("Int64")
# recreate the eval_inputs you used for score_dag (if you built it by merging)
eval_inputs = ds.files.merge(scans[["id"]].drop_duplicates(), on="id", how="inner")


# # intersection check
# sc_ids = set(scans["id"].dropna().astype(int)) if "id" in scans.columns else set()
# ds_ids = set(ds.files["id"].dropna().astype(int))
# print("counts -> scans:", len(sc_ids), "ds.files:", len(ds_ids), "intersection:", len(sc_ids & ds_ids))
# print("example ids only in scans (up to 10):", sorted(list(sc_ids - ds_ids))[:10])
# print("example ids only in ds.files (up to 10):", sorted(list(ds_ids - sc_ids))[:10])

##############################################################################

# delete openai in env if present, we have a local model
for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
    os.environ.pop(k, None)

# should be moved to env
os.environ["VLLM_SERVER_BASE_URL"] = "http://127.0.0.1:8001/v1"
# dspy seems to fall back to openai in case of errors, this is a dummy key
os.environ["OPENAI_API_KEY"] = "sk-local"

lm_provider_mod = importlib.reload(lm_provider_mod)
fe = importlib.reload(fe)

# replace vision lm with preprocessed scans
pdf_to_text.VLLM_Preprocessing = lambda **kw: LoadScans(CACHE_CSV)

training = importlib.reload(training)


_base_parts = training.get_training_dag()

expected_final_pipeline = infering.build_complete_inference_dag(_base_parts)
expected_final_pipeline


ArchivingDateProvider.predict = _predict_safe

# INTERVENTION_DATE extractor

# replace intervention start extractor to not crash when context is too large
InterventionStartExtractor._to_dspy_input = _to_dspy_input_patched


ide._get_min_date = _get_min_date_safe
ide._get_max_date = _get_max_date_safe


# COMUNE extractor

_COMUNI, _MERGED = _read_comuni_tables_strict()
_POS2ID = _COMUNI.index.to_list()

_FIELDS = set(Comune.model_fields.keys())

# replace comune extractor code here because the original didn't work
ComuneExtractor._to_dspy_input = _comune_to_dspy_input


def _esecutore_to_dspy_input(self, x):
    return EsecutoreInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
        # possibili_esecutori=_as_list(getattr(x, "possibili_esecutori", [])),
    )

EsecutoreExtractor._to_dspy_input = _esecutore_to_dspy_input


def _protocollo_to_dspy_input(self, x):
    return ProtocolloInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

ProtocolloExtractor._to_dspy_input = _protocollo_to_dspy_input


def _tipo_to_dspy_input(self, x):
    return TipoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

TipoExtractor._to_dspy_input = _tipo_to_dspy_input


def _ogd_to_dspy_input(self, x):
    return TipoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

OGDExtractor._to_dspy_input = _ogd_to_dspy_input


def _luogo_to_dspy_input(self, x):
    return LuogoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

LuogoExtractor._to_dspy_input = _luogo_to_dspy_input


def _year_to_dspy_input(self, x):
    return DataInterventoInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )

YearExtractor._to_dspy_input = _year_to_dspy_input


def _direzione_funzionario_to_dspy_input(self, x):
    return DirezioneFunzionarioInputData(
        fragmenti_relazione=getattr(x, "merged_chunks", ""),
    )
DirezioneFunzionarioExtractor._to_dspy_input = _direzione_funzionario_to_dspy_input

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


RitrovamentoExtractor._to_dspy_input = _ritrovamenti_to_dspy_input


with mlflow.start_run():
    trained_dag_parts = training.train_from_scratch(train_inputs, ds)
    per_field_scores, detailed_results = infering.score_dag(trained_dag_parts, eval_inputs, ds)
    
    # Save as CSV
    detailed_results.to_csv("detailed_results.csv", index=False)
    
    # Log as MLflow artifact
    mlflow.log_artifact("detailed_results.csv")


# # Save as CSV
# detailed_results.to_csv("detailed_results2.csv", index=False)

# # Log as MLflow artifact
# mlflow.log_artifact("detailed_results.csv")


visualizator.init_complete_vizualisation_engine(detailed_results)


# run display server (blocks)
visualizator.run_display_server()
