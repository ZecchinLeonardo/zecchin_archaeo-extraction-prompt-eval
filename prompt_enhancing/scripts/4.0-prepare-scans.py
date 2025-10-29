#!/usr/bin/env python3
"""Converted from notebook: 4.0-prepare-scans.ipynb

This script mirrors the notebook cells in order. It attempts to enable
IPython autoreload when run in an interactive IPython environment.

Note: Running the full script requires the project's runtime environment
and dependencies (VLLM, archaeo_super_prompt, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).resolve().parents[1]  # .../prompt_enhancing
src_dir = str(project_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from archaeo_super_prompt.utils.cache import get_cache_dir_for
from archaeo_super_prompt.dataset import MagohDataset
from archaeo_super_prompt.modeling.pdf_to_text import VLLM_Preprocessing


# def enable_autoreload() -> None:
#     """Enable IPython autoreload if running inside IPython.

#     This mirrors the notebook magics:
#       %load_ext autoreload
#       %autoreload 2
#     """
#     try:
#         from IPython import get_ipython

#         ip = get_ipython()
#         if ip is not None:
#             # run_line_magic expects (magic_name, arg)
#             ip.run_line_magic("load_ext", "autoreload")
#             ip.run_line_magic("autoreload", "2")
#     except Exception:
#         # Not running in IPython or IPython not present; ignore
#         pass


# def main() -> None:
    # enable_autoreload()

    # --- Cell: imports ---
    
    # --- Cell: scanner setup ---

csv_path = Path(__file__).resolve().parent / "media.csv"

scanner = VLLM_Preprocessing(
    vlm_provider="vllm",
    # vlm_model_id="ibm-granite/granite-vision-3.3-2b",
    vlm_model_id="google/gemma-3-27b-it",
    incipit_only=True,
    prompt="OCR this part of Italian document for markdown-based processing.",
    embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
)


try:
    # media.csv is semicolon-separated; read as strings to avoid dtype issues, convert later.
    df = pd.read_csv(csv_path, sep=';', engine='python', encoding='utf-8', dtype=str)
except Exception as e:
    raise SystemExit(f"Failed to read CSV {csv_path}: {e}")

# ensure ruolo is numeric (or compare to string '8' if appropriate)
df['ruolo'] = pd.to_numeric(df['ruolo'], errors='coerce')

# keep rows where ruolo == 8
mask = df['ruolo'] == 8

# extract the digits after the last '/' in area, convert to int and get unique list
area_ids = (
    df.loc[mask, 'area']
    .astype(str)
    .str.extract(r'/(\d+)$')[0]        # use r'/(\d+)$' for the last slash; remove $ if not last
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

selected_ids = area_ids

ds = MagohDataset(selected_ids)
inputs = ds.files

# --- Cell: display sample (converted to print) ---
# try:
#     # Try to print a small slice safely; guard if inputs isn't the expected type
#     print("Sample inputs (first 83 rows):")
#     print(getattr(inputs, 'iloc', lambda *a, **k: inputs)(:83))
# except Exception:
#     # Fallback: print repr
#     print(repr(inputs)[:1000])

# --- Cell: env prints ---
print("VLM_HOST_URL:", os.environ.get("VLM_HOST_URL"))
print("VLLM_SERVER_BASE_URL:", os.environ.get("VLLM_SERVER_BASE_URL"))
print("OLLAMA_SERVER_BASE_URL:", os.environ.get("OLLAMA_SERVER_BASE_URL"))

# --- Cell: run transform in chunks ---
chunk_size = 10
print(f"Starting scanner.transform in chunks of {chunk_size} documents...")

# prepare output path for per-chunk appends
cache_dir = get_cache_dir_for("interim", "miscel")
cache_dir.mkdir(parents=True, exist_ok=True)
out_path = Path(cache_dir) / "scans_vllm.csv"

# Read existing scanned ids (if any) so we can skip them
existing_ids = set()
existing_row_count = 0
first_write = not out_path.exists()
if out_path.exists():
    try:
        existing_df = pd.read_csv(out_path)
        if 'id' in existing_df.columns:
            existing_ids = set(existing_df['id'].astype(str).tolist())
            existing_row_count = len(existing_df)
            print(f"Found {len(existing_ids)} already-scanned ids in {out_path} (rows: {existing_row_count})")
    except Exception as e:
        print(f"Warning: could not read existing scans CSV {out_path}: {e}")

outputs = []
total = None
try:
    total = len(inputs)
except Exception:
    # if inputs doesn't support len(), try to get via index
    try:
        total = inputs.shape[0]
    except Exception:
        total = None

if total is None:
    # Fallback: try single transform call
    print("Could not determine number of input rows; attempting single transform call")
    try:
        output = scanner.transform(inputs)
        # Ensure dataframe and skip already existing ids before writing
        try:
            if not isinstance(output, pd.DataFrame):
                output = pd.DataFrame(output)
        except Exception:
            pass

        # If ids present, filter out ones already scanned
        try:
            if isinstance(output, pd.DataFrame) and 'id' in output.columns:
                mask_new = ~output['id'].astype(str).isin(existing_ids)
                new_output = output.loc[mask_new]
            else:
                new_output = output
        except Exception as excf:
            print(f"Warning while filtering existing ids for single-call output: {excf}")
            new_output = output

        # append this single output immediately to file (if any rows remain)
        try:
            if isinstance(new_output, pd.DataFrame) and len(new_output) == 0:
                print("Single-call output contains no new rows after filtering existing ids; skipping write.")
            else:
                write_header = first_write
                # assign a running integer index that continues from existing_row_count
                start_index = existing_row_count
                try:
                    new_output.index = range(start_index, start_index + len(new_output))
                except Exception:
                    pass

                if hasattr(new_output, "to_csv"):
                    # write including the unnamed index column so the CSV matches the notebook output
                    if write_header:
                        new_output.to_csv(out_path, mode='a', header=True, index=True)
                    else:
                        new_output.to_csv(out_path, mode='a', header=False, index=True)
                    print(f"Appended single-call output to {out_path} (header_written={write_header})")
                else:
                    odf = pd.DataFrame(new_output)
                    if write_header:
                        odf.to_csv(out_path, mode='a', header=True, index=True)
                    else:
                        odf.to_csv(out_path, mode='a', header=False, index=True)
                    print(f"Appended single-call converted output to {out_path} (header_written={write_header})")

                # update bookkeeping
                try:
                    if isinstance(new_output, pd.DataFrame) and 'id' in new_output.columns:
                        new_ids = set(new_output['id'].astype(str).tolist())
                        existing_ids.update(new_ids)
                        existing_row_count += len(new_output)
                except Exception:
                    pass

                first_write = False

        except Exception as exc:
            print(f"Failed to append single-call output to CSV: {exc}")
    except Exception as exc:
        print("scanner.transform failed on full inputs:", exc)
else:
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        try:
            subset = inputs.iloc[start:end]
        except Exception:
            # Fall back to positional slicing if necessary
            try:
                subset = inputs[start:end]
            except Exception as exc:
                print(f"Failed to slice inputs for chunk {start}-{end}: {exc}")
                continue

        print(f"Processing chunk {start}..{end - 1} (size={len(subset)})")
        # Filter subset to skip already-existing ids before scanning
        try:
            if hasattr(subset, 'columns') and 'id' in subset.columns:
                mask_new = ~subset['id'].astype(str).isin(existing_ids)
                subset_to_scan = subset.loc[mask_new]
            else:
                subset_to_scan = subset
        except Exception:
            subset_to_scan = subset

        if getattr(subset_to_scan, 'shape', (None,))[0] == 0 or len(subset_to_scan) == 0:
            print(f"Chunk {start}-{end-1}: all rows already scanned or no rows to scan, skipping")
            continue

        try:
            out_chunk = scanner.transform(subset_to_scan)
            # append the (filtered) chunk to outputs for later concat/inspection
            outputs.append(out_chunk)
            try:
                n_rows = len(out_chunk)
            except Exception:
                n_rows = 'unknown'
            print(f"Chunk processed, produced {n_rows} rows")
            # append chunk to CSV immediately to avoid keeping everything in memory
            try:
                # convert to DataFrame if necessary
                if not isinstance(out_chunk, pd.DataFrame):
                    try:
                        out_chunk = pd.DataFrame(out_chunk)
                    except Exception:
                        pass

                # If ids are present in the output, filter out any that are already present (safety)
                try:
                    if isinstance(out_chunk, pd.DataFrame) and 'id' in out_chunk.columns:
                        mask_new_out = ~out_chunk['id'].astype(str).isin(existing_ids)
                        out_chunk = out_chunk.loc[mask_new_out]
                except Exception:
                    pass

                if isinstance(out_chunk, pd.DataFrame) and len(out_chunk) == 0:
                    print(f"After filtering existing ids, chunk {start}-{end-1} has no new rows to write; skipping write.")
                else:
                    write_header = first_write

                    # assign a running integer index that continues from existing_row_count
                    start_index = existing_row_count
                    try:
                        out_chunk.index = range(start_index, start_index + len(out_chunk))
                    except Exception:
                        pass

                    if hasattr(out_chunk, "to_csv"):
                        # write including the unnamed index column so the CSV matches the notebook output
                        if write_header:
                            out_chunk.to_csv(out_path, mode='a', header=True, index=True)
                        else:
                            out_chunk.to_csv(out_path, mode='a', header=False, index=True)
                        print(f"Appended chunk to {out_path} (header_written={write_header})")
                    else:
                        cdf = pd.DataFrame(out_chunk)
                        if write_header:
                            cdf.to_csv(out_path, mode='a', header=True, index=True)
                        else:
                            cdf.to_csv(out_path, mode='a', header=False, index=True)
                        print(f"Appended converted chunk to {out_path} (header_written={write_header})")

                    # update bookkeeping
                    try:
                        if isinstance(out_chunk, pd.DataFrame) and 'id' in out_chunk.columns:
                            new_ids = set(out_chunk['id'].astype(str).tolist())
                            existing_ids.update(new_ids)
                            existing_row_count += len(out_chunk)
                    except Exception:
                        pass

                    first_write = False
            except Exception as exc:
                print(f"Failed to append chunk to CSV: {exc}")
        except Exception as exc:
            print(f"scanner.transform failed for chunk {start}-{end}: {exc}")
            # continue with next chunk

# --- Cell: inspect and save concatenated output ---
if outputs:
    try:
        # Try to concat as DataFrames
        output = pd.concat(outputs, ignore_index=True)
    except Exception:
        # If concat fails, keep first non-empty output as representative
        output = None
        for o in outputs:
            if o is not None:
                output = o
                break

    if output is not None:
        try:
            unique_count = len(output["id"].unique())
            print("Unique id count in output:", unique_count)
        except Exception:
            print("Could not compute unique id count; output type/shape unexpected")

        # Save to cache as in notebook
        try:
            # If we already appended chunks to out_path, avoid writing the full output again
            if out_path.exists():
                print(f"Final output already written/accumulated at {out_path}; skipping full save.")
            else:
                # If not appended for some reason, write the full output once
                if hasattr(output, "to_csv"):
                    output.to_csv(out_path)
                    print(f"Output saved to {out_path}")
                else:
                    # try saving repr
                    with open(out_path, "w", encoding="utf-8") as fh:
                        fh.write(repr(output))
                    print(f"Output saved (repr) to {out_path}")
        except Exception as exc:
            print("Failed to save output to CSV:", exc)
else:
    print("No outputs were produced by chunked processing.")

