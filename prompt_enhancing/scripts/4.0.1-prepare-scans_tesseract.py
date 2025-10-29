# Converted from notebook: exploratory/4.0-prepare-scans.ipynb
# Notes:
# - IPython magics (%load_ext, %autoreload) are guarded and replaced with imports where appropriate.
# - This script assumes it's run with the project's Python environment where `archaeo_super_prompt` is importable.

import os
import sys
import pandas as pd
from pathlib import Path


# # Try to mimic %load_ext autoreload; if IPython is available enable autoreload for interactive runs
# try:
#     from IPython import get_ipython
#     ip = get_ipython()
#     if ip is not None:
#         try:
#             ip.run_line_magic("load_ext", "autoreload")
#             ip.run_line_magic("autoreload", "2")
#         except Exception:
#             # ignore if magics are not applicable
#             pass
# except Exception:
#     pass


project_root = Path(__file__).resolve().parents[1]  # .../prompt_enhancing
src_dir = str(project_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
# Save output to cache directory using the project's helper
from archaeo_super_prompt.utils.cache import get_cache_dir_for

from archaeo_super_prompt.dataset import MagohDataset
from archaeo_super_prompt.utils.tesseract_ocr import TesseractPreprocessing

# from archaeo_super_prompt.modeling.pdf_to_text import VLLM_Preprocessing

# # Build scanner (same parameters as notebook)
# scanner = VLLM_Preprocessing(
#     vlm_provider="vllm",
#     vlm_model_id="ibm-granite/granite-vision-3.3-2b",
#     incipit_only=True,
#     prompt="OCR this part of Italian document for markdown-based processing.",
#     embedding_model_hf_id="nomic-ai/nomic-embed-text-v1.5",
# )

scanner = TesseractPreprocessing(lang="ita", incipit_only=False)

# # Selected document ids from notebook
# _digitally_born_documents = [
#     # very good
#     33799, 34439, 38005, 36837, 36937, 37614, 37026, 37971, 36846, 36304, 34423, 36052,
#     37043, 36554, 989, 37007, 30897, 36351, 36308, 38013, 36011, 33828, 1221,
#     38039, 35429, 37065, 37116, 34452, 33441, 33062, 34939, 35918, 33689, 34508, 31035,
#     38220, 38092, 36979, 36854, 36207, 34915, 35688, 36359,
#     # not that good
#     31164, 32600, 33760, 32714, 31208, 30712,
# ]

# ok_scanned_pdfs = {
#     32666, 31298, 33548, 35189, 35399, 30925, 37040, 37379, 33589, 34769,
#     33858, 34329, 5193, 37706, 30647, 37702, 33540, 36042, 33357, 34959,
#     30646, 33547, 32581, 30878, 37302, 33560, 35881, 31031, 37381, 242, 34869,
#     33841, 36465, 33499, 36095, 36068, 33594, 33904, 33644, 33553, 35052,
#     33630, 34426, 31090, 30716, 31059, 35849, 33813, 34666, 36119, 830,
#     36187, 31977, 34787, 33749, 35447, 33555, 33846, 34093, 33508,  # 33710
# }

# dirty_pdfs = {
#     36648, 32433, 35131, 33383, 30657, 31312, 30399, 33331, 31234, 30548,
#     34685, 34237, 35114, 30821, 33708, 33668, 34932, 30697, 38241, 33443,
#     37305, 33535, 31815, 35203, 33576, 32053, 33761, 37910, 35983, 31314,
#     37400, 36457, 33582, 31903, 32494, 33184, 36070, 31804, 30861
# }

# selected_ids = set(_digitally_born_documents).union(ok_scanned_pdfs, dirty_pdfs)

# df = pd.read_csv("./media.csv")
csv_path = Path(__file__).resolve().parent / "media.csv"

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
print(selected_ids)

ds = MagohDataset(selected_ids)
inputs = ds.files

# In the notebook they show inputs.iloc[:83] so we can print a small preview
try:
    print("Preview of inputs (first 10 rows):")
    # inputs may be a pandas DataFrame or similar
    print(getattr(inputs, 'head', lambda n: inputs[:n])(10))
except Exception:
    # fallback
    try:
        print(list(inputs)[:10])
    except Exception:
        pass

# Process inputs in chunks and append each chunk's result to CSV
chunk_size = 10

cache_dir = get_cache_dir_for("interim", "miscel")
cache_dir.mkdir(parents=True, exist_ok=True)
out_path = cache_dir / "scans_tesseract.csv"

# Determine number of inputs
n_inputs = None
try:
    n_inputs = len(inputs)
except Exception:
    try:
        inputs = pd.DataFrame(inputs)
        n_inputs = len(inputs)
    except Exception:
        raise SystemExit("Could not determine number of inputs for chunking")

print(f"Processing {n_inputs} inputs in chunks of {chunk_size}...")

# Read existing scanned ids if scans.csv exists and skip them
existing_ids = set()
# track number of existing rows so we can continue the unnamed leading index
existing_row_count = 0
if out_path.exists():
    try:
        existing_df = pd.read_csv(out_path)
        if 'id' in existing_df.columns:
            existing_ids = set(existing_df['id'].astype(str).tolist())
            existing_row_count = len(existing_df)
            print(f"Found {len(existing_ids)} already-scanned ids in {out_path} (rows: {existing_row_count})")
    except Exception as e:
        print(f"Warning: could not read existing scans CSV {out_path}: {e}")

first_write = not out_path.exists()
ids_seen = set()

for start in range(0, n_inputs, chunk_size):
    end = min(start + chunk_size, n_inputs)
    try:
        chunk = inputs.iloc[start:end]
    except Exception:
        chunk = pd.DataFrame(inputs)[start:end]

    # If chunk has an 'id' column, filter out already scanned rows
    if 'id' in chunk.columns:
        mask_new = ~chunk['id'].astype(str).isin(existing_ids)
        chunk_to_scan = chunk[mask_new]
    else:
        chunk_to_scan = chunk

    if len(chunk_to_scan) == 0:
        print(f"Chunk {start}-{end-1}: all rows already scanned, skipping")
        continue

    print(f"Scanning chunk {start}-{end-1}: scanning {len(chunk_to_scan)} rows...")
    try:
        out_chunk = scanner.transform(chunk_to_scan)
    except Exception as e:
        print(f"Scanner failed on chunk {start}-{end-1}: {e}")
        continue

    # Convert to DataFrame if needed
    if not isinstance(out_chunk, pd.DataFrame):
        try:
            out_chunk = pd.DataFrame(out_chunk)
        except Exception as e:
            print(f"Could not convert scanner output to DataFrame for chunk {start}-{end-1}: {e}")
            continue

    # Update seen/existing ids from the output
    if 'id' in out_chunk.columns:
        new_ids = set(out_chunk['id'].astype(str).tolist())
        ids_seen.update(new_ids)
        existing_ids.update(new_ids)

    # Minimal validation: expect the scanner to return the canonical columns.
    desired_cols = [
        'id', 'filename', 'chunk_type', 'chunk_page_position',
        'chunk_index', 'chunk_embedding_content', 'chunk_content'
    ]

    missing = [c for c in desired_cols if c not in out_chunk.columns]
    if missing:
        print(f"Warning: scanner output missing columns {missing}; filling defaults for them.")
        # fill sensible defaults for missing columns
        if 'filename' in missing:
            out_chunk['filename'] = ''
        if 'chunk_type' in missing:
            out_chunk['chunk_type'] = [['text']] if len(out_chunk) > 0 else []
        if 'chunk_page_position' in missing:
            out_chunk['chunk_page_position'] = [[0] for _ in range(len(out_chunk))]
        if 'chunk_index' in missing:
            out_chunk['chunk_index'] = list(range(len(out_chunk)))
        if 'chunk_embedding_content' in missing:
            out_chunk['chunk_embedding_content'] = ''
        if 'chunk_content' in missing:
            out_chunk['chunk_content'] = ''

    # Reorder to canonical columns (any extra columns are dropped)
    out_chunk = out_chunk.reindex(columns=desired_cols)

    # Write/appends
    try:
        # assign a running integer index that continues from existing_row_count
        start_index = existing_row_count
        out_chunk.index = range(start_index, start_index + len(out_chunk))

        if first_write:
            # write header and include the index (unnamed leading column)
            out_chunk.to_csv(out_path, index=True)
            first_write = False
        else:
            # append without header but include the index so the leading column continues
            out_chunk.to_csv(out_path, index=True, header=False, mode='a')

        # update the running count so future chunks continue numbering
        existing_row_count += len(out_chunk)
        print(f"Wrote chunk {start}-{end-1} ({len(out_chunk)} rows) to {out_path} (rows now: {existing_row_count})")
    except Exception as e:
        print(f"Failed to write chunk {start}-{end-1} to CSV: {e}")

print(f"Done. New ids written: {len(ids_seen)}. Total scanned ids now: {len(existing_ids)}")

# End of script
