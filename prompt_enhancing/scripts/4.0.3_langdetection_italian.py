import os
import sys
import pandas as pd
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]  # .../prompt_enhancing
src_dir = str(project_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
# Save output to cache directory using the project's helper
from archaeo_super_prompt.utils.cache import get_cache_dir_for
from archaeo_super_prompt.utils.language_discriminator import ItalianDiscriminator, VLLMItalianDiscriminator
from archaeo_super_prompt.modeling import pdf_to_text


# cache location
csv_file = input("Enter the name of the scans CSV file (default: 'scans.csv'): ") or "scans.csv"
CACHE_CSV = get_cache_dir_for("interim", "miscel") / csv_file

# --- CHANGES: initialize discriminator with the chosen cache CSV and run checks ---
# create discriminator bound to this cache CSV
# detection = ItalianDiscriminator(CACHE_CSV)

detection = VLLMItalianDiscriminator(
    cache_csv=CACHE_CSV,
    italian_threshold=0.75,
    max_chunks_check=3,
    vllm_url="http://localhost:8001",
    model=None,
    timeout=20,
)


def _annotate_df(df: pd.DataFrame, disc: ItalianDiscriminator) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=(list(df.columns) if df is not None else []) + ["italian_score", "is_italian"])
    out = df.copy()
    for col in ("chunk_embedding_content", "chunk_content", "filename"):
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    scores = []
    flags = []
    for _, r in out.iterrows():
        sample = ""
        for c in ("chunk_embedding_content", "chunk_content"):
            if c in r and str(r.get(c, "")).strip():
                sample = str(r.get(c, "")).strip()
                break
        sc = float(disc._is_italian_score(sample))
        scores.append(sc)
        flags.append(sc >= disc.italian_threshold)
    out["italian_score"] = scores
    out["is_italian"] = flags
    return out

def _try_vllm_rescan(filename: str) -> pd.DataFrame:
    """
    Try to obtain fresh chunks for `filename` using project's VLLM preprocessor.
    Returns a DataFrame or empty DataFrame on failure.
    """
    VClass = getattr(pdf_to_text, "VLLM_Preprocessing", None)
    if VClass is None:
        return pd.DataFrame()
    try:
        # best-effort params from env or defaults
        vlm_provider = os.getenv("VLM_PROVIDER", "vllm")
        vlm_model_id = os.getenv("VLM_MODEL_ID", "local-vlm")
        prompt = os.getenv("VLM_PROMPT", "Extract the textual content of this PDF.")
        embedding_model_hf_id = os.getenv("EMBEDDING_MODEL_HF_ID", "sentence-transformers/all-MiniLM-L6-v2")
        incipit_only = bool(int(os.getenv("VLM_INCPIT_ONLY", "1")))
        inst = VClass(
            vlm_provider=vlm_provider,
            vlm_model_id=vlm_model_id,
            prompt=prompt,
            embedding_model_hf_id=embedding_model_hf_id,
            incipit_only=incipit_only,
        )
        X = pd.DataFrame([{"id": filename, "filepath": filename}])
        out = inst.transform(X)
        if isinstance(out, pd.DataFrame):
            return out
        try:
            return pd.DataFrame(out)
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def main():
    try:
        detection.fit()
    except Exception as e:
        print(f"Failed to load scans CSV '{CACHE_CSV}': {e}")
        return

    df = detection._df
    if df is None or df.empty:
        print("No rows found in cache CSV.")
        return

    # decide grouping key: prefer 'id' if present, otherwise 'filename'
    group_key = "id" if "id" in df.columns else "filename"

    # prepare output CSV next to the cache CSV and write header
    out_path = CACHE_CSV.with_name(CACHE_CSV.stem + "_italian_check.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # remove existing file if present so we recreate it freshly
    if out_path.exists():
        try:
            out_path.unlink()
        except OSError as e:
            print(f"Unable to remove existing file {out_path}: {e}")
            return

    # fixed columns for per-chunk CSV
    csv_columns = [
        "id",
        "filename",
        "chunk_index",
        "chunk_page_position",
        "original_chunks",
        "returned_chunks",
        "rescanned",
        "italian_score",
        "is_italian",
    ]
    # write header
    pd.DataFrame(columns=csv_columns).to_csv(out_path, index=False)

    # FINAL file that receives italian chunks and rescanned chunks; when rescan fails append an EMPTY chunk
    final_path = CACHE_CSV.with_name(CACHE_CSV.stem + "_final_chunks.csv")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        try:
            final_path.unlink()
        except OSError as e:
            print(f"Unable to remove existing file {final_path}: {e}")
            return

    final_cols = list(df.columns)
    for c in ("chunk_index", "chunk_page_position", "chunk_embedding_content", "chunk_content"):
        if c not in final_cols:
            final_cols.append(c)
    final_cols += ["italian_score", "is_italian", "rescanned_source"]
    pd.DataFrame(columns=final_cols).to_csv(final_path, index=False)

    # iterate groups and emit one row per chunk with per-chunk scores
    total_rows = 0
    italians = 0
    results = []
    for key_value, group in df.groupby(group_key, sort=False):
        # determine filename and doc_id for processing
        if group_key == "id":
            doc_id = key_value
            # try to get a filename from the group (may be empty)
            filename = group["filename"].astype(str).replace("nan", "").iloc[0] if "filename" in group.columns else ""
        else:
            doc_id = None
            filename = key_value

        # call process_file (this may return cached rows or rescanned rows)
        try:
            out_rows = detection.process_file(filename=filename, doc_id=doc_id)
        except Exception:
            out_rows = pd.DataFrame()

        original_count = int(len(group))
        out_count = int(len(out_rows))
        rescanned = out_count != original_count

        # For each chunk returned, record a row with per-chunk italian score
        if out_rows is None or out_rows.empty:
            row = {
                "id": str(key_value) if group_key == "id" else "",
                "filename": filename,
                "chunk_index": "",
                "chunk_page_position": "",
                "original_chunks": original_count,
                "returned_chunks": out_count,
                "rescanned": rescanned,
                "italian_score": 0.0,
                "is_italian": False,
            }
            pd.DataFrame([row])[csv_columns].to_csv(out_path, mode="a", header=False, index=False)
            total_rows += 1
            if row["is_italian"]:
                italians += 1

            results.append({
                "id": str(key_value) if group_key == "id" else "",
                "filename": filename,
                "original_chunks": original_count,
                "returned_chunks": out_count,
                "rescanned": rescanned,
                "italian_chunks": 0,
                "is_italian": False,
            })
            # append an empty chunk to final file when nothing is returned
            empty_row = {c: "" for c in final_cols}
            empty_row.update({"id": str(key_value) if group_key == "id" else "", "filename": filename,
                              "italian_score": 0.0, "is_italian": False, "rescanned_source": "rescan_failed_empty"})
            pd.DataFrame([empty_row])[final_cols].to_csv(final_path, mode="a", header=False, index=False)
            continue

        # ensure annotation exists
        if "is_italian" not in out_rows.columns or "italian_score" not in out_rows.columns:
            out_rows = _annotate_df(out_rows, detection)

        # append per-chunk italian_check rows
        for _, r in out_rows.iterrows():
            chunk_index = r.get("chunk_index", "") if "chunk_index" in out_rows.columns else ""
            chunk_page = r.get("chunk_page_position", "") if "chunk_page_position" in out_rows.columns else ""
            excerpt = ""
            if "chunk_embedding_content" in out_rows.columns and str(r.get("chunk_embedding_content", "")).strip():
                excerpt = str(r.get("chunk_embedding_content", "")).strip()
            elif "chunk_content" in out_rows.columns and str(r.get("chunk_content", "")).strip():
                excerpt = str(r.get("chunk_content", "")).strip()
            excerpt = (excerpt[:200] + "...") if len(excerpt) > 200 else excerpt
            italian_score = float(r.get("italian_score", 0.0))
            is_chunk_italian = bool(r.get("is_italian", italian_score >= detection.italian_threshold))

            row = {
                "id": str(key_value) if group_key == "id" else "",
                "filename": filename,
                "chunk_index": chunk_index,
                "chunk_page_position": chunk_page,
                "original_chunks": original_count,
                "returned_chunks": out_count,
                "rescanned": rescanned,
                "italian_score": italian_score,
                "is_italian": is_chunk_italian,
            }
            pd.DataFrame([row])[csv_columns].to_csv(out_path, mode="a", header=False, index=False)
            total_rows += 1
            if row["is_italian"]:
                italians += 1

        italians_df = out_rows[out_rows["is_italian"].astype(bool)].copy()
        non_ital_df = out_rows[~out_rows["is_italian"].astype(bool)].copy()

        # append italian chunks to final file
        if not italians_df.empty:
            italians_df = italians_df.copy()
            italians_df["rescanned_source"] = "cached_rescanned" if rescanned else "cached"
            for c in final_cols:
                if c not in italians_df.columns:
                    italians_df[c] = ""
            italians_df[final_cols].to_csv(final_path, mode="a", header=False, index=False)

        # for non-italian chunks: attempt to re-scan and append re-scanned chunks
        if not non_ital_df.empty:
            rescanned_df = _try_vllm_rescan(filename)
            if rescanned_df is None or rescanned_df.empty:
                # rescan failed: append an EMPTY chunk row for each original non-italian chunk,
                # preserving available metadata (chunk_index, chunk_page_position, ...)
                rows_to_append = []
                for _, nr in non_ital_df.iterrows():
                    empty_row = {c: "" for c in final_cols}
                    # copy any metadata present in the original chunk row
                    for col in nr.index:
                        if col in final_cols:
                            try:
                                empty_row[col] = nr.get(col, "") if pd.notnull(nr.get(col, "")) else ""
                            except Exception:
                                empty_row[col] = ""
                    empty_row.update({
                        "id": str(key_value) if group_key == "id" else "",
                        "filename": filename,
                        "italian_score": 0.0,
                        "is_italian": False,
                        "rescanned_source": "rescan_failed_empty",
                    })
                    rows_to_append.append(empty_row)
                if not rows_to_append:
                    # defensive fallback: single fully empty row
                    fallback = {c: "" for c in final_cols}
                    fallback.update({
                        "id": str(key_value) if group_key == "id" else "",
                        "filename": filename,
                        "italian_score": 0.0,
                        "is_italian": False,
                        "rescanned_source": "rescan_failed_empty",
                    })
                    rows_to_append.append(fallback)
                pd.DataFrame(rows_to_append)[final_cols].to_csv(final_path, mode="a", header=False, index=False)
            else:
                rescann_annot = _annotate_df(rescanned_df, detection)
                rescann_annot["rescanned_source"] = "vllm_rescan"
                for c in final_cols:
                    if c not in rescann_annot.columns:
                        rescann_annot[c] = ""
                rescann_annot[final_cols].to_csv(final_path, mode="a", header=False, index=False)
        
        try:
            italian_chunks = int(out_rows["is_italian"].astype(bool).sum())
        except Exception:
            italian_chunks = 0

        # append document-level summary
        doc_summary = {
            "id": str(key_value) if group_key == "id" else "",
            "filename": filename,
            "original_chunks": original_count,
            "returned_chunks": out_count,
            "rescanned": rescanned,
            "italian_chunks": 0,
            "is_italian": False,
        }
        results.append(doc_summary)

    #     else:
    #         for _, r in out_rows.iterrows():
    #             # pick chunk index/page if present
    #             chunk_index = r.get("chunk_index", "") if "chunk_index" in out_rows.columns else ""
    #             chunk_page = r.get("chunk_page_position", "") if "chunk_page_position" in out_rows.columns else ""
    #             # prefer embedding content for excerpt, else chunk_content
    #             excerpt = ""
    #             if "chunk_embedding_content" in out_rows.columns and str(r.get("chunk_embedding_content", "")).strip():
    #                 excerpt = str(r.get("chunk_embedding_content", "")).strip()
    #             elif "chunk_content" in out_rows.columns and str(r.get("chunk_content", "")).strip():
    #                 excerpt = str(r.get("chunk_content", "")).strip()
    #             # truncate excerpt for CSV readability
    #             excerpt = (excerpt[:200] + "...") if len(excerpt) > 200 else excerpt

    #             italian_score = float(r.get("italian_score", 0.0)) if "italian_score" in out_rows.columns else float(detection._is_italian_score(excerpt))
    #             is_chunk_italian = bool(r.get("is_italian", italian_score >= detection.italian_threshold))

    #             row = {
    #                 "id": str(key_value) if group_key == "id" else "",
    #                 "filename": filename,
    #                 "chunk_index": chunk_index,
    #                 "chunk_page_position": chunk_page,
    #                 # "chunk_excerpt": excerpt,
    #                 "original_chunks": original_count,
    #                 "returned_chunks": out_count,
    #                 "rescanned": rescanned,
    #                 "italian_score": italian_score,
    #                 "is_italian": is_chunk_italian,
    #             }
    #             pd.DataFrame([row])[csv_columns].to_csv(out_path, mode="a", header=False, index=False)
    #             total_rows += 1
    #             if row["is_italian"]:
    #                 italians += 1

    # # compute document-level italian summary from out_rows if available
    #         if out_rows is not None and not out_rows.empty and "is_italian" in out_rows.columns:
    #             try:
    #                 italian_chunks = int(out_rows["is_italian"].astype(bool).sum())
    #             except Exception:
    #                 italian_chunks = 0
    #         else:
    #             italian_chunks = 0

    #         doc_summary = {
    #             "id": str(key_value) if group_key == "id" else "",
    #             "filename": filename,
    #             "original_chunks": original_count,
    #             "returned_chunks": out_count,
    #             "rescanned": rescanned,
    #             "italian_chunks": italian_chunks,
    #             "is_italian": bool(italian_chunks > 0),
    #         }
    #         results.append(doc_summary)

    # write document-level summary CSV next to the cache CSV (keep per-chunk CSV intact)
    summary_path = CACHE_CSV.with_name(CACHE_CSV.stem + "_italian_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(results).to_csv(summary_path, index=False)
    except Exception as e:
        print(f"Failed to write summary CSV '{summary_path}': {e}")
        return

    # brief console summary
    total = len(results)
    italian_docs = sum(1 for r in results if r.get("is_italian"))
    print(f"Processed {total} documents. {italian_docs} appear to be Italian (threshold={detection.italian_threshold}).")
    print(f"Per-chunk details written to: {out_path}")
    print(f"Document summary written to: {summary_path}")

if __name__ == "__main__":
    main()