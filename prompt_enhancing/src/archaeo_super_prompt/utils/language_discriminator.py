from pathlib import Path
from typing import Optional
import re

import pandas as pd

import requests

from sklearn.base import BaseEstimator, TransformerMixin

from langdetect import detect_langs

# adapt to your project import path
from archaeo_super_prompt.modeling import pdf_to_text


class ItalianDiscriminator(BaseEstimator, TransformerMixin):
    """
    Load a scans CSV and decide per-document whether its OCR looks Italian.
    If not Italian, attempt to re-scan the PDF using the project's VLLM preprocessor.

    Usage:
      disc = ItalianDiscriminator(cache_csv="path/to/scans.csv")
      disc.fit()
      df_for_doc = disc.process_file(filename=".../doc.pdf", doc_id=123)

    Behavior:
      - Only the first `max_chunks_check` non-empty chunks (preferring
        chunk_embedding_content then chunk_content) are sampled to decide language.
      - If langdetect is installed it will be used; otherwise a small heuristic
        based on common Italian function words and accented characters is applied.
      - If re-scanning is attempted it calls the original pdf_to_text.VLLM_Preprocessing
        (if available) and returns its DataFrame when successful.
    """

    def __init__(self, cache_csv: str | Path, italian_threshold: float = 0.6, max_chunks_check: int = 3):
        self.cache_csv = Path(cache_csv)
        self.italian_threshold = float(italian_threshold)
        self.max_chunks_check = int(max_chunks_check)
        self._df: Optional[pd.DataFrame] = None
        # keep reference to any original VLLM preprocessor
        self._orig_vllm = getattr(pdf_to_text, "VLLM_Preprocessing", None)

    def fit(self, X=None, y=None):
        self._df = self._load_scans_safe(self.cache_csv)
        return self

    def _load_scans_safe(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # normalize common columns used by this class
        if "id" in df.columns:
            try:
                df["id"] = df["id"].astype("Int64")
            except Exception:
                df["id"] = df["id"].astype(str)
        for col in ("chunk_embedding_content", "chunk_content", "filename"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df

    def _is_italian_score(self, text: str) -> float:
        txt = (text or "").strip()
        if not txt:
            return 0.0
        # prefer langdetect if available
        try:
            langs = detect_langs(txt[:4000])
            for p in langs:
                if p.lang == "it":
                    return float(min(1.0, p.prob))
            return 0.0
        except Exception:
            # fallback heuristic
            txt_l = txt.lower()
            words = re.findall(r"\w+", txt_l)
            if not words:
                return 0.0
            italian_common = {
                "di",
                "e",
                "il",
                "la",
                "che",
                "in",
                "per",
                "con",
                "una",
                "un",
                "non",
                "si",
                "del",
                "della",
                "lo",
                "al",
            }
            count_common = sum(1 for w in words[:200] if w in italian_common)
            score_common = count_common / max(1, min(200, len(words)))
            accents = sum(1 for ch in txt if ch in "àèéìíòóùú")
            score_accents = min(1.0, accents / 3.0)
            return min(1.0, 0.7 * score_common + 0.3 * score_accents)

    def _concat_first_chunks(self, rows: pd.DataFrame) -> str:
        texts = []
        # sort rows (try to use chunk_index / chunk_page_position if available)
        sort_cols = [c for c in ("chunk_index", "chunk_page_position") if c in rows.columns]
        if sort_cols:
            rows = rows.sort_values(by=sort_cols, na_position="last")
        # prefer embedding content first
        for col in ("chunk_embedding_content", "chunk_content"):
            if col not in rows.columns:
                continue
            for t in rows[col].astype(str).tolist():
                if t and len(texts) < self.max_chunks_check:
                    texts.append(t)
                if len(texts) >= self.max_chunks_check:
                    break
            if len(texts) >= self.max_chunks_check:
                break
        return " ".join(texts)

    def _select_first_chunk_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with at most self.max_chunks_check rows, choosing rows
        with non-empty chunk_embedding_content or chunk_content after sorting.
        """
        if rows is None or rows.empty:
            return rows

        # sort rows (same logic as in _concat_first_chunks)
        sort_cols = [c for c in ("chunk_index", "chunk_page_position") if c in rows.columns]
        if sort_cols:
            rows = rows.sort_values(by=sort_cols, na_position="last")

        selected_idx = []
        cols_pref = ("chunk_embedding_content", "chunk_content")
        for idx, row in rows.iterrows():
            # pick rows that have any non-empty preferred content
            has_content = False
            for c in cols_pref:
                if c in rows.columns:
                    val = str(row.get(c, "") or "").strip()
                    if val:
                        has_content = True
                        break
            if has_content:
                selected_idx.append(idx)
            if len(selected_idx) >= self.max_chunks_check:
                break

        # fallback: if we didn't find any with content, take the first N rows
        if not selected_idx:
            return rows.head(self.max_chunks_check).copy()

        return rows.loc[selected_idx].copy()

    def process_file(self, filename: str, doc_id: Optional[int] = None) -> pd.DataFrame:
        """
        Return DataFrame rows for the document (cached or re-scanned).
        If doc_id is provided, prefer matching by id; otherwise try by filename.
        """
        if self._df is None:
            self.fit()
        cand = pd.DataFrame()
        if doc_id is not None and "id" in self._df.columns:
            cand = self._df[self._df["id"].astype(str) == str(doc_id)]
        else:
            # try endswith match then exact filename match
            fname_series = self._df["filename"].astype(str)
            cand = self._df[fname_series.str.endswith(str(filename))]
            if cand.empty:
                cand = self._df[fname_series == str(filename)]

        def _annotate_rows(rows: pd.DataFrame) -> pd.DataFrame:
            # ensure we have string columns
            if rows is None or rows.empty:
                # create empty frame with expected columns
                cols = list(self._df.columns) if self._df is not None else []
                cols = cols + ["italian_score", "is_italian"]
                return pd.DataFrame(columns=cols)

            df = rows.copy()
            for col in ("chunk_embedding_content", "chunk_content", "filename"):
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)

            def _pick_sample(r: pd.Series) -> str:
                for c in ("chunk_embedding_content", "chunk_content"):
                    if c in r and str(r[c]).strip():
                        return str(r[c]).strip()
                return ""

            scores = []
            flags = []
            for _, row in df.iterrows():
                sample = _pick_sample(row)
                score = float(self._is_italian_score(sample))
                scores.append(score)
                flags.append(score >= self.italian_threshold)

            df["italian_score"] = scores
            df["is_italian"] = flags

            # NOTE: score ALL chunks for the document (user requested full per-chunk
            # reporting). Do not trim here; caller can later pick a subset if needed.
            return df

        # if no cached rows, try vllm preproc immediately
        if cand.empty:
            if callable(self._orig_vllm):
                try:
                    out = self._orig_vllm(filename=filename)
                    if isinstance(out, pd.DataFrame):
                        return _annotate_rows(out)
                except Exception:
                    pass
            # return empty frame with same columns if possible
            return _annotate_rows(pd.DataFrame())

        # annotate cached rows (we still compute per-chunk scores)
        annotated = _annotate_rows(cand)

        # if the majority / sample of chunks already looks italian we can return them
        # (we still return per-chunk flags so caller sees which chunks are Italian)
        if not annotated.empty:
            # if sample concatenation suggests italian we keep cached (but annotated)
            # else attempt re-scan and prefer its annotated rows if available
            sample_text = self._concat_first_chunks(cand)
            score = self._is_italian_score(sample_text)
            if score >= self.italian_threshold:
                return annotated

        # attempt to re-scan using original vllm preprocessor (prefer its result)
        if callable(self._orig_vllm):
            try:
                res = self._orig_vllm(filename=filename)
                if isinstance(res, pd.DataFrame) and not res.empty:
                    # sanitize and annotate re-scan rows
                    for col in ("chunk_embedding_content", "chunk_content", "filename"):
                        if col in res.columns:
                            res[col] = res[col].fillna("").astype(str)
                    res_annot = _annotate_rows(res)
                    if not res_annot.empty:
                        return res_annot
            except Exception:
                pass

        # fallback to annotated cached OCR
        return annotated


class VLLMItalianDiscriminator(ItalianDiscriminator):
    """
    Variant of ItalianDiscriminator that asks a running vllm server to classify
    the language/confidence for a given text chunk sample.

    Usage:
      disc = VLLMItalianDiscriminator(cache_csv="scans.csv",
                                      vllm_url="http://localhost:8001",
                                      model="google/gemma-3-27b-it",
                                      italian_threshold=0.6,
                                      max_chunks_check=3)
      disc.fit()
      df_for_doc = disc.process_file(filename=".../doc.pdf")
    """

    def __init__(self, cache_csv: str | Path, italian_threshold: float = 0.6,
                 max_chunks_check: int = 3, vllm_url: str = "http://localhost:8001",
                 model: Optional[str] = None, timeout: int = 20):
        super().__init__(cache_csv=cache_csv, italian_threshold=italian_threshold,
                         max_chunks_check=max_chunks_check)
        self.vllm_url = str(vllm_url).rstrip("/")
        self.model = model  # optional model name for the vllm endpoint payload
        self.timeout = int(timeout)

    def _call_vllm(self, prompt: str) -> str:
        """
        Try several common vllm/serve endpoints and return the textual output.
        This is intentionally permissive about response shapes and uses the raw
        text body as fallback.
        """
        if not prompt:
            return ""

        endpoints = [
            f"{self.vllm_url}/v1/completions",
            f"{self.vllm_url}/v1/generate",
            f"{self.vllm_url}/generate",
        ]
        headers = {"Content-Type": "application/json"}
        # small, deterministic prompt asking for JSON output
        body_prompt = (
            "Determine the language of the following text and return a JSON object "
            'with keys "language" (e.g. "it") and "confidence" (0.0-1.0). '
            "Respond only with the JSON object.\n\nText:\n'''\n" + prompt + "\n'''"
        )

        for ep in endpoints:
            try:
                if ep.endswith("/v1/completions"):
                    payload = {"prompt": body_prompt, "max_tokens": 32, "temperature": 0.0}
                    if self.model:
                        payload["model"] = self.model
                    resp = requests.post(ep, json=payload, headers=headers, timeout=self.timeout)
                else:
                    payload = {"prompt": body_prompt, "max_tokens": 32, "temperature": 0.0}
                    if self.model:
                        payload["model"] = self.model
                    resp = requests.post(ep, json=payload, headers=headers, timeout=self.timeout)

                if not resp.ok:
                    continue

                # try best-effort parsing of common JSON shapes
                try:
                    j = resp.json()
                    # OpenAI-like: choices[0].text
                    if isinstance(j, dict):
                        if "choices" in j and j["choices"]:
                            first = j["choices"][0]
                            if isinstance(first, dict):
                                text = first.get("text") or first.get("message", {}).get("content")
                                if text:
                                    return str(text).strip()
                        # vllm may return an 'output' or 'outputs' structure
                        for k in ("output", "outputs", "result"):
                            if k in j:
                                val = j[k]
                                if isinstance(val, str):
                                    return val.strip()
                                if isinstance(val, list) and val:
                                    return str(val[0]).strip()
                    # fallback to raw text
                except Exception:
                    pass

                # fallback: raw body
                return resp.text.strip()
            except Exception:
                continue

        # if all endpoints fail, raise so caller can fallback
        raise RuntimeError(f"Failed to reach vllm at {self.vllm_url}")

    def _parse_vllm_jsonish(self, out: str) -> tuple[str, float]:
        """
        Try to extract language and confidence from the model output text.
        Returns (language_code or '', confidence [0-1]).
        """
        if not out:
            return "", 0.0

        # try to find JSON-like fields "language" and "confidence"
        lang_m = re.search(r'"language"\s*:\s*"([^"]+)"', out, re.IGNORECASE)
        conf_m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', out, re.IGNORECASE)
        if lang_m:
            lang = lang_m.group(1).strip().lower()
            conf = float(conf_m.group(1)) if conf_m else 1.0
            return lang, max(0.0, min(1.0, conf))

        # try simple text patterns like: Italian (0.95) or it:0.9
        m = re.search(r'([A-Za-z]+)\s*\(?\s*([0-9]*\.?[0-9]+)\s*\)?', out)
        if m:
            lang = m.group(1).strip().lower()
            try:
                conf = float(m.group(2))
                if conf > 1 and conf <= 100:
                    conf = conf / 100.0
                return lang, max(0.0, min(1.0, conf))
            except Exception:
                return lang, 1.0

        # final best-effort: look for the word 'ital' or 'italiano' or 'it'
        if re.search(r'\bital(?:ian|iano)?\b', out, re.IGNORECASE):
            return "it", 1.0

        return "", 0.0

    def _is_italian_score(self, text: str) -> float:
        """
        Override to ask the vllm model for a language + confidence.
        Falls back to parent's implementation on any failure.
        """
        txt = (text or "").strip()
        if not txt:
            return 0.0

        # limit prompt size
        try:
            out = self._call_vllm(txt[:2000])
            lang, conf = self._parse_vllm_jsonish(out)
            if not lang:
                # fallback to parent heuristic/detect
                return super()._is_italian_score(text)
            # normalize language codes/names
            if lang in ("it", "ita", "italian", "italiano"):
                return float(conf)
            # not italian
            return 0.0
        except Exception:
            return super()._is_italian_score(text)