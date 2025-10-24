# Replace scanner with a Tesseract-based scanner

import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

# For computing chunk_embedding_content in the same way as chunking.chunk_to_ds
try:
    from archaeo_super_prompt.modeling.pdf_to_text.chunking import (
        get_chunker,
        EMBED_MODEL_ID,
    )
except Exception:
    # Defensive: if import fails, we'll fallback to using raw chunk text as embedding content
    get_chunker = None
    EMBED_MODEL_ID = None

class TesseractPreprocessing:
    """Tesseract-based preprocessing that returns a chunk-style dataset compatible
    with the rest of the pipeline (PDFChunkDataset-like).

    The transform method returns a pandas.DataFrame with columns:
    id, filename, chunk_type, chunk_page_position, chunk_index,
    chunk_embedding_content, chunk_content
    """

    def __init__(self, lang='ita', incipit_only=True, max_lines=10, max_chunk_tokens: int = 512):
        self.lang = lang
        self.incipit_only = incipit_only
        self.max_lines = max_lines
        self.max_chunk_tokens = max_chunk_tokens

    def _image_to_text(self, img):
        return pytesseract.image_to_string(img, lang=self.lang)

    def transform(self, df, pages=None):
        """Read files from the DataFrame `df` and produce one chunk per page.

        Parameters
        - df: DataFrame with at least an `id` and a path-like column (path/file/filepath/filename)
        - pages: optional number of pages to read per document (overrides incipit_only)
        """

        records = []

        # initialize a chunker/tokenizer to compute embedding content similarly to
        # chunking.chunk_to_ds -> chunker.contextualize(chunk)
        chunker = None
        tokenizer = None
        if get_chunker is not None and EMBED_MODEL_ID is not None:
            try:
                chunker = get_chunker(EMBED_MODEL_ID, self.max_chunk_tokens)
                # access underlying HF tokenizer if available
                tokenizer = getattr(getattr(chunker, "tokenizer", None), "tokenizer", None)
            except Exception:
                chunker = None
                tokenizer = None

        for _, row in df.iterrows():
            # find path
            path = None
            for key in ("path", "file", "filepath", "filename"):
                if key in row and pd.notna(row[key]):
                    path = Path(row[key])
                    break
            if path is None:
                candidate = row.iloc[0] if len(row) > 0 else None
                if isinstance(candidate, (str, Path)):
                    path = Path(candidate)
            if path is None:
                continue

            file_id = row.get("id", None)
            filename = path.name

            try:
                if path.suffix.lower() == ".pdf":
                    # determine which pages to read
                    if pages is not None:
                        pages_imgs = convert_from_path(str(path), first_page=1, last_page=pages)
                    elif self.incipit_only:
                        pages_imgs = convert_from_path(str(path), first_page=1, last_page=1)
                    else:
                        pages_imgs = convert_from_path(str(path))

                    # iterate pages and create one chunk per page
                    for idx, img in enumerate(pages_imgs):
                        page_no = idx + 1
                        text = self._image_to_text(img)
                        if self.incipit_only and page_no == 1 and self.max_lines:
                            lines = [l for l in text.splitlines() if l.strip()]
                            text = "\n".join(lines[: self.max_lines])

                        # compute embedding content: try to mimic chunker.contextualize
                        emb_content = ""
                        try:
                            if chunker is not None and hasattr(chunker, "contextualize"):
                                # Best-effort: create a tiny pseudo-chunk with .text attribute
                                class _Tmp:
                                    def __init__(self, text):
                                        self.text = text

                                # chunker.contextualize expects a BaseChunk; many implementations
                                # only use chunk.text internally. Try to call it safely.
                                try:
                                    emb_content = chunker.contextualize(_Tmp(text))
                                except Exception:
                                    emb_content = None
                            elif tokenizer is not None:
                                # fallback: tokenize+decode with truncation to approximate
                                toks = tokenizer(text, truncation=True, max_length=self.max_chunk_tokens)
                                emb_content = tokenizer.decode(toks["input_ids"], skip_special_tokens=True)
                            else:
                                emb_content = None
                        except Exception:
                            emb_content = None

                        # Ensure we never leave embedding blank: fallback to raw text
                        if not emb_content:
                            emb_content = text

                        records.append(
                            {
                                "id": file_id,
                                "filename": filename,
                                "chunk_type": ["text"],
                                "chunk_page_position": [page_no],
                                "chunk_index": idx,
                                "chunk_embedding_content": emb_content,
                                "chunk_content": text,
                            }
                        )
                else:
                    img = Image.open(path)
                    text = self._image_to_text(img)
                    if self.incipit_only and self.max_lines:
                        lines = [l for l in text.splitlines() if l.strip()]
                        text = "\n".join(lines[: self.max_lines])
                    # compute embedding content for single-image files
                    emb_content = ""
                    try:
                        if chunker is not None and hasattr(chunker, "contextualize"):
                            class _Tmp:
                                def __init__(self, text):
                                    self.text = text

                            try:
                                emb_content = chunker.contextualize(_Tmp(text))
                            except Exception:
                                emb_content = None
                        elif tokenizer is not None:
                            toks = tokenizer(text, truncation=True, max_length=self.max_chunk_tokens)
                            emb_content = tokenizer.decode(toks["input_ids"], skip_special_tokens=True)
                        else:
                            emb_content = None
                    except Exception:
                        emb_content = None

                    if not emb_content:
                        emb_content = text

                    records.append(
                        {
                            "id": file_id,
                            "filename": filename,
                            "chunk_type": ["text"],
                            "chunk_page_position": [1],
                            "chunk_index": 0,
                            "chunk_embedding_content": emb_content,
                            "chunk_content": text,
                        }
                    )
            except Exception as e:
                records.append(
                    {
                        "id": file_id,
                        "filename": filename,
                        "chunk_type": ["text"],
                        "chunk_page_position": [0],
                        "chunk_index": 0,
                        "chunk_embedding_content": "",
                        "chunk_content": f"[OCR ERROR: {e}]",
                    }
                )

        # Return a DataFrame with the canonical column order
        df_out = pd.DataFrame.from_records(records)
        desired = [
            'id', 'filename', 'chunk_type', 'chunk_page_position',
            'chunk_index', 'chunk_embedding_content', 'chunk_content'
        ]
        for c in desired:
            if c not in df_out.columns:
                df_out[c] = ''

        return df_out[desired]