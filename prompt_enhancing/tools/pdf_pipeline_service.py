"""PDF pipeline service

Provides a small FastAPI app exposing an endpoint to POST a PDF file.

Steps performed:
- save uploaded PDF to a temporary file
- run Tesseract OCR chunking (using project `TesseractPreprocessing`)
- detect language for each chunk (using `langdetect`)
- if any chunk is not Italian, re-run OCR for that chunk with VLLM preprocessing
- write a CSV of Italian chunks in order to the project cache dir
- optionally invoke the existing `4.1.1_complete_pipeline.py` script and feed it the CSV filename

Note: This script depends on project modules and the `langdetect` package.
Run with: `uvicorn prompt_enhancing.tools.pdf_pipeline_service:app --port 9000`
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import tempfile
import uuid
import os
import sys
from pathlib import Path
import shutil
import subprocess
import logging
import pandas as pd
import threading
import json
import time
import ast


import importlib

from archaeo_super_prompt.dataset import MagohDataset
from archaeo_super_prompt.utils.pipeline_func import _comune_to_dspy_input, _esecutore_to_dspy_input, _protocollo_to_dspy_input, _tipo_to_dspy_input, _ogd_to_dspy_input, _luogo_to_dspy_input, _year_to_dspy_input, _direzione_funzionario_to_dspy_input, _ritrovamenti_to_dspy_input


app = FastAPI(title="PDF OCR+Extract pipeline")

logger = logging.getLogger("pdf_pipeline")
logging.basicConfig(level=logging.INFO)

# Try to import project OCR/VLLM helpers
try:
    from archaeo_super_prompt.utils.tesseract_ocr import TesseractPreprocessing
except Exception:
    TesseractPreprocessing = None

try:
    from archaeo_super_prompt.modeling.pdf_to_text import VLLM_Preprocessing
except Exception:
    VLLM_Preprocessing = None


# Require the project's VLLM-based discriminator for language detection
try:
    from archaeo_super_prompt.utils.language_discriminator import VLLMItalianDiscriminator
except Exception:
    VLLMItalianDiscriminator = None

from archaeo_super_prompt.utils.cache import get_cache_dir_for

from archaeo_super_prompt.utils.load_scans import LoadScans
from archaeo_super_prompt.utils.pipeline_func import *



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


# instantiate a discriminator if available (we don't call fit here because
# we will work with per-request DataFrames; discriminator's _is_italian_score
# is useful standalone and VLLMItalianDiscriminator will call the vllm server
# when used).
if VLLMItalianDiscriminator is None:
    raise RuntimeError(
        "VLLMItalianDiscriminator not available. Ensure `archaeo_super_prompt.utils.language_discriminator` is importable."
    )

# instantiate the VLLM-based discriminator
vllm_url = os.getenv("VLLM_SERVER_BASE_URL", os.getenv("VLLM_URL", "http://127.0.0.1:8001"))
_DISC = VLLMItalianDiscriminator(
    cache_csv=Path(tempfile.gettempdir()) / "_dummy_scans_cache.csv",
    italian_threshold=float(os.getenv("ITALIAN_THRESHOLD", "0.7")),
    # max_chunks_check=int(os.getenv("MAX_CHUNKS_CHECK", "3")),
    vllm_url=vllm_url,
    model=os.getenv("VLLM_MODEL", None),
    timeout=int(os.getenv("VLLM_TIMEOUT", "20")),
)

# Job directory for async extraction
JOB_DIR = Path(tempfile.gettempdir()) / "pdf_pipeline_jobs"
JOB_DIR.mkdir(parents=True, exist_ok=True)

def _clean_value(v):
    # convert pandas/NaN to None and try to parse list-like strings
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    # if it's a string that looks like a list/dict, try ast.literal_eval
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # common case: '[]' or "['a','b']"
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
    return v


def _clean_df_to_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.items():
            # normalize key: strip, lower, replace spaces
            key = str(k).strip()
            key = key.replace(" ", "_").lower()
            rec[key] = _clean_value(v)
        out.append(rec)
    return out


def _locate_and_load_detailed_results(repo_root: Path, job_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Try multiple strategies to locate and load detailed_results.csv.

    Returns (df, error_message). If df is found returns (df, None), else (None, errmsg).
    """
    # 1) direct path in repo_root/src
    try_paths = []
    try_paths.append(repo_root / "src" / "detailed_results.csv")

    # 2) local mlruns under repo_root
    try:
        for p in repo_root.rglob("detailed_results.csv"):
            try_paths.append(p)
    except Exception:
        pass

    # 3) common mlruns locations (cwd and system temp)
    for base in (Path.cwd(), Path(tempfile.gettempdir())):
        try:
            for p in base.rglob("detailed_results.csv"):
                try_paths.append(p)
        except Exception:
            pass

    # pick the most recently modified candidate if any
    candidates = [p for p in try_paths if p.exists()]
    if candidates:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            return pd.read_csv(chosen), None
        except Exception as e:
            return None, f"found CSV at {chosen} but failed to read: {e}"

    # We intentionally do not attempt to query MLflow here. Consumers should
    # prefer the filesystem outputs that the in-process runner writes into
    # the job directory. If nothing is found on disk, report not found.
    return None, "detailed_results.csv not found in repo or local mlruns"


def run_tesseract_on_pdf(pdf_path: Path):
    if TesseractPreprocessing is None:
        raise RuntimeError("TesseractPreprocessing not available from project. Check imports.")
    scanner = TesseractPreprocessing(lang="ita", incipit_only=False)
    # Build a dataframe with one row as expected by scanner.transform
    inputs = pd.DataFrame([{"id": 1, "filename": str(pdf_path)}])
    out = scanner.transform(inputs)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out)
    return out


def run_vllm_on_chunks(chunks_df: pd.DataFrame):
    if VLLM_Preprocessing is None:
        raise RuntimeError("VLLM_Preprocessing not available from project. Check imports.")
    # instantiate a vllm preprocessor similar to the project's training config
    vllm_proc = VLLM_Preprocessing(
        vlm_provider="vllm",
        vlm_model_id="ibm-granite/granite-vision-3.3-2b",
        incipit_only=False,
        prompt="OCR this part of Italian document for markdown-based processing.",
        embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
    )
    out = vllm_proc.transform(chunks_df)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out)
    return out


def is_italian(text: str, min_prob: float = 0.7) -> bool:
    """Return True if detected language is Italian using VLLMItalianDiscriminator.

    This function calls the project's VLLM-based discriminator which queries the
    running vllm server. It raises if the discriminator isn't available.
    """
    if not text or not isinstance(text, str):
        return False

    if _DISC is None:
        raise RuntimeError("VLLMItalianDiscriminator is not initialized")

    try:
        score = float(_DISC._is_italian_score(text))
        return score >= float(min_prob)
    except Exception as e:
        # propagate a clear error to the caller
        raise RuntimeError(f"VLLM language detection failed: {e}")


@app.post("/extract_pdf")
async def extract_pdf(file: UploadFile = File(...), run_extraction_script: bool = True, scan_id: int | None = Form(None)):
    """Upload a PDF, OCR it, ensure Italian chunks, write CSV and optionally run extraction script.

    Returns JSON with the CSV path and extraction stdout/stderr if run.
    """
    # Basic checks
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    tmpd = Path(tempfile.gettempdir()) / "pdf_pipeline_tmp"
    tmpd.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    pdf_path = tmpd / f"upload_{uid}.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    pdf_id = None
    if scan_id is not None:
        # scan_id is received as an integer form field; prefer it
        pdf_id = str(int(scan_id))
    

    # final fallback to uid if nothing numeric found
    if pdf_id is None:
        # we do not allow non-numeric ids to propagate into training, require user input
        raise HTTPException(status_code=400, detail=("scan_id not provided or not extractable from filename; "+
                                                     "please provide a numeric scan_id as a form field: -F 'scan_id=34083'"))

    logger.info(f"Saved uploaded PDF to {pdf_path}")

    # 1) run tesseract OCR chunking on whole document
    try:
        scanned = run_tesseract_on_pdf(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tesseract processing failed: {e}")

    # Ensure canonical columns exist
    for col in ("chunk_content", "chunk_index", "chunk_page_position"):
        if col not in scanned.columns:
            scanned[col] = None

    # 2) detect language per chunk
    scanned = scanned.reset_index(drop=True)
    scanned["_is_italian"] = scanned["chunk_content"].fillna("").apply(lambda t: is_italian(t))

    # 3) re-read non-italian chunks with vllm
    non_it_mask = ~scanned["_is_italian"].astype(bool)
    replacements = {}
    if non_it_mask.any():
        # Build a DataFrame to pass to VLLM_Preprocessing. We try to preserve page position
        to_rerun = scanned.loc[non_it_mask, ["id", "filename", "chunk_page_position"]].copy()
        # VLLM_Preprocessing expects the same interface as scanner.transform: a DataFrame of files.
        # Ensure filename column is present; if not fallback to the saved pdf path.
        if "filename" not in to_rerun.columns or to_rerun["filename"].isnull().all():
            to_rerun["filename"] = str(pdf_path)
        try:
            vllm_out = run_vllm_on_chunks(to_rerun)
        except Exception as e:
            logger.warning(f"VLLM re-read failed: {e}")
            vllm_out = pd.DataFrame()

        # If we got results, try to map them back by chunk_index or page position
        if not vllm_out.empty:
            # prefer columns 'chunk_index' or 'chunk_page_position' to map
            for _, row in vllm_out.iterrows():
                key = None
                if "chunk_index" in row and not pd.isna(row["chunk_index"]):
                    key = int(row["chunk_index"])
                elif "chunk_page_position" in row and not pd.isna(row["chunk_page_position"]):
                    # page position may be list-like; try to use first element
                    val = row["chunk_page_position"]
                    try:
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            key = int(val[0])
                        else:
                            key = int(val)
                    except Exception:
                        key = None
                if key is not None and "chunk_content" in row:
                    replacements[key] = row["chunk_content"]

    # 4) produce final italian chunks in order (prefer replaced content)
    def final_chunk_content(idx, original_text):
        if idx in replacements:
            return replacements[idx]
        return original_text

    if "chunk_index" in scanned.columns and scanned["chunk_index"].notna().any():
        scanned = scanned.sort_values(by="chunk_index")
        scanned["final_content"] = [final_chunk_content(int(r.get("chunk_index", i)), r.get("chunk_content", "")) for i, r in scanned.iterrows()]
    else:
        scanned["final_content"] = scanned.apply(lambda r: final_chunk_content(r.name, r.get("chunk_content", "")), axis=1)

    # keep only Italian chunks (either originally Italian or replaced by vllm)
    def kept(r):
        if int(r.name) in replacements:
            # replacement assumed to be Italian
            return True
        return bool(r.get("_is_italian", False))

    scanned["_keep"] = scanned.apply(kept, axis=1)
    final_chunks = scanned.loc[scanned["_keep"], :].copy()

    # write CSV in cache folder
    cache_dir = get_cache_dir_for("interim", "miscel")
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"scan_{uid}_tesseract.csv"
    csv_path = cache_dir / csv_name

    # canonical columns and requested CSV layout
    out_cols = ["id", "filename", "chunk_type", "chunk_page_position", "chunk_index", "chunk_embedding_content", "final_content"]
    for c in out_cols:
        if c not in final_chunks.columns:
            final_chunks[c] = ""

    final_out = final_chunks[out_cols].rename(columns={"final_content": "chunk_content"})

    # ensure every row has a usable filename: training preprocessing expects
    # a non-empty 'filename' column (pandera validates it). If the scanner
    # didn't provide filenames, fall back to the uploaded PDF path.
    try:
        if (
            "filename" not in final_out.columns
            or final_out["filename"].isnull().all()
            or (final_out["filename"].astype(str).str.strip() == "").all()
        ):
            final_out["filename"] = str(pdf_path)
            logger.info(f"Populated missing 'filename' in scan CSV with uploaded PDF path {pdf_path}")
    except Exception:
        # be lenient: if any check fails, proceed to write CSV anyway
        pass

    # set the 'id' for every row to the determined pdf_id (as integer)
    try:
        final_out["id"] = int(str(pdf_id))
    except Exception:
        raise HTTPException(status_code=400, detail="determined scan id is not numeric; provide numeric scan_id via form field")

    # ensure the DataFrame has a progressive unnamed index starting at 0
    final_out = final_out.reset_index(drop=True)
    final_out.index = range(len(final_out))

    # write CSV including the index as the first unnamed column
    final_out.to_csv(csv_path, index=True)
    logger.info(f"Wrote italian chunks CSV to {csv_path}")

    result = {"csv_path": str(csv_path), "replacements_made": len(replacements)}

    if run_extraction_script:
        job_id = uuid.uuid4().hex
        job_dir = JOB_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"

        # create a status file before starting
        status_file = job_dir / "status.json"
        status_file.write_text(json.dumps({"job_id": job_id, "status": "running", "start_ts": time.time()}))

        def _run_pipeline_inthread(cache_csv_name: str, jd: Path, repo_root: Path):
            try:
                # redirect stdout/stderr into files
                with open(stdout_path, "wb") as sof, open(stderr_path, "wb") as sef:
                    text_out = sof
                    text_err = sef
                    old_cwd = Path.cwd()
                    try:
                        # switch to repo src (the notebook did this). Be robust:
                        # search upward from the provided repo_root and cwd for a folder containing `src`.
                        def _find_src_dir(start: Path | None):
                            if start is None:
                                return None
                            for anc in (start, ) + tuple(start.parents):
                                cand = anc / "src"
                                if cand.exists() and cand.is_dir():
                                    return cand
                            return None

                        # compute cache CSV full path
                        cache_csv = get_cache_dir_for("interim", "miscel") / cache_csv_name
                        # load scans CSV and dataset
                        try:
                            SCANS_DF = pd.read_csv(cache_csv)
                        except Exception as e:
                            text_err.write(f"failed to read scans CSV {cache_csv}: {e}\n".encode())
                            raise

                        # build dataset
                        selected_ids = set(map(int, SCANS_DF["id"].dropna().tolist()))
                        ds = MagohDataset(selected_ids)

                        inputs = ds.files.merge(SCANS_DF[["id"]].drop_duplicates(), on="id", how="inner")
                        # simple split similar to the notebook
                        train_inputs, eval_inputs = inputs.iloc[:10], inputs.iloc[10:]

                        # robustly reload training/predict modules
                        try:
                            training = importlib.reload(importlib.import_module("archaeo_super_prompt.modeling.train"))
                        except Exception:
                            training = importlib.import_module("archaeo_super_prompt.modeling.train")
                        try:
                            infering = importlib.reload(importlib.import_module("archaeo_super_prompt.modeling.predict"))
                        except Exception:
                            infering = importlib.import_module("archaeo_super_prompt.modeling.predict")

                        # recompute eval_inputs using load_scans_safe if available
                        try:
                            # from archaeo_super_prompt.utils.pipeline_func import load_scans_safe
                            scans = load_scans_safe(cache_csv)
                            ds.files["id"] = pd.to_numeric(ds.files["id"].astype(str).str.strip(), errors="coerce").astype("Int64")
                            eval_inputs = ds.files.merge(scans[["id"]].drop_duplicates(), on="id", how="inner")

                            # delete openai in env if present, we have a local model
                            for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
                                os.environ.pop(k, None)

                            # should be moved to env
                            os.environ["VLLM_SERVER_BASE_URL"] = "http://127.0.0.1:8001/v1"
                            # dspy seems to fall back to openai in case of errors, this is a dummy key
                            os.environ["OPENAI_API_KEY"] = "sk-local"

                            # lm_provider_mod = importlib.reload(lm_provider_mod)
                            # fe = importlib.reload(fe)

                            # # replace vision lm with preprocessed scans
                            # pdf_to_text.VLLM_Preprocessing = lambda **kw: LoadScans(CACHE_CSV)

                            training = importlib.reload(training)


                            _base_parts = training.get_training_dag()

                            expected_final_pipeline = infering.build_complete_inference_dag(_base_parts)
                            expected_final_pipeline


                            ComuneExtractor._to_dspy_input = _comune_to_dspy_input
                            EsecutoreExtractor._to_dspy_input = _esecutore_to_dspy_input
                            ProtocolloExtractor._to_dspy_input = _protocollo_to_dspy_input
                            TipoExtractor._to_dspy_input = _tipo_to_dspy_input
                            OGDExtractor._to_dspy_input = _ogd_to_dspy_input
                            LuogoExtractor._to_dspy_input = _luogo_to_dspy_input
                            YearExtractor._to_dspy_input = _year_to_dspy_input
                            DirezioneFunzionarioExtractor._to_dspy_input = _direzione_funzionario_to_dspy_input
                            RitrovamentoExtractor._to_dspy_input = _ritrovamenti_to_dspy_input




                            trained_dag_parts = training.train_from_scratch(train_inputs, ds)
                            per_field_scores, detailed_results = infering.score_dag(trained_dag_parts, eval_inputs, ds)
    
                            # Save as CSV
                            detailed_results.to_csv("detailed_results.csv", index=False)

                        except Exception:
                            # fall back to previously computed eval_inputs
                            raise
                            # pass

                        # run training -> inference similar to the notebook
                        try:
                            try:
                                text_out.write(b"About to start training/inference\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        
                        returncode = 0

                        # update status and write result.json
                        try:
                            st = json.loads(status_file.read_text()) if status_file.exists() else {}
                        except Exception:
                            st = {}
                        st.update({"status": "finished", "returncode": returncode, "end_ts": time.time()})
                        status_file.write_text(json.dumps(st))

                        # attempt to locate/load detailed_results.csv and serialize
                        df, err = _locate_and_load_detailed_results(repo_root, jd)
                        if df is not None:
                            try:
                                records = _clean_df_to_records(df)
                                (jd / "result.json").write_text(json.dumps({"extracted_data": records}, ensure_ascii=False))
                                # also save a CSV copy of the extracted results into the job folder
                                try:
                                    out_csv_result = jd / "result.csv"
                                    df.to_csv(out_csv_result, index=False)
                                except Exception:
                                    out_csv_result = None
                                # also copy CSV into project results cache folder for easier discovery
                                try:
                                    results_cache_base = get_cache_dir_for("interim", "miscel")
                                    results_dir = results_cache_base / "results"
                                    results_dir.mkdir(parents=True, exist_ok=True)
                                    if out_csv_result is not None and out_csv_result.exists():
                                        out_csv_repo = results_dir / f"result_{jd.name}.csv"
                                        shutil.copyfile(out_csv_result, out_csv_repo)
                                        # append saved path to status
                                        try:
                                            st = json.loads(status_file.read_text()) if status_file.exists() else {}
                                        except Exception:
                                            st = {}
                                        st.update({"saved_result_csv": str(out_csv_repo)})
                                        status_file.write_text(json.dumps(st))
                                except Exception:
                                    pass
                            except Exception as e:
                                (jd / "result.json").write_text(json.dumps({"error": f"failed to clean/serialize CSV: {e}"}))
                        else:
                            (jd / "result.json").write_text(json.dumps({"error": err}))

                    finally:
                        os.chdir(old_cwd)
            except Exception as e:
                try:
                    status_file.write_text(json.dumps({"status": "failed", "error": str(e)}))
                except Exception:
                    pass

        # repo root is two parents up from this file
        repo_root = Path(__file__).resolve().parents[2]
        # start worker thread
        t = threading.Thread(target=_run_pipeline_inthread, args=(csv_name, job_dir, repo_root), daemon=True)
        t.start()

        result["job_id"] = job_id
        result["job_status"] = "running"

    return JSONResponse(result)



@app.get("/extract_status/{job_id}")
def extract_status(job_id: str):
    """Get job status and result if available."""
    job_dir = JOB_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job_id not found")
    status_file = job_dir / "status.json"
    result_file = job_dir / "result.json"
    resp = {"job_id": job_id}
    if status_file.exists():
        try:
            resp.update(json.loads(status_file.read_text()))
        except Exception:
            resp["status_read_error"] = "failed to read status.json"
    else:
        resp["status"] = "unknown"

    if result_file.exists():
        try:
            resp.update(json.loads(result_file.read_text()))
        except Exception:
            resp["result_read_error"] = "failed to read result.json"

    return JSONResponse(resp)


if __name__ == "__main__":
    # quick launcher for local testing
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
