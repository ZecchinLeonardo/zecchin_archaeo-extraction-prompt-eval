from pathlib import Path
from typing import Optional, Tuple
import re

import requests

# reuse existing ItalianDiscriminator for selection/processing helpers
from archaeo_super_prompt.utils.language_discriminator import ItalianDiscriminator


class VLLMItalianDiscriminator(ItalianDiscriminator):
    """
    VLLM-backed variant of ItalianDiscriminator.

    - Queries a running vllm serve (default http://localhost:8001) to ask the model
      to determine language and a confidence score for a text sample.
    - Falls back to the parent's langdetect/heuristic on any failure.
    """

    def __init__(
        self,
        cache_csv: str | Path,
        italian_threshold: float = 0.6,
        max_chunks_check: int = 3,
        vllm_url: str = "http://localhost:8001",
        model: Optional[str] = None,
        timeout: int = 20,
    ):
        super().__init__(cache_csv=cache_csv, italian_threshold=italian_threshold, max_chunks_check=max_chunks_check)
        self.vllm_url = str(vllm_url).rstrip("/")
        self.model = model
        self.timeout = int(timeout)

    def _call_vllm(self, prompt: str) -> str:
        """
        Try several common vllm endpoints and return the raw textual response.
        Raises RuntimeError if all endpoints fail.
        """
        if not prompt:
            return ""

        endpoints = [
            f"{self.vllm_url}/v1/completions",
            f"{self.vllm_url}/v1/generate",
            f"{self.vllm_url}/generate",
        ]
        headers = {"Content-Type": "application/json"}
        body_prompt = (
            "Determine the language of the following text and return a JSON object "
            'with keys \"language\" (e.g. \"it\") and \"confidence\" (0.0-1.0). '
            "Respond only with the JSON object.\n\nText:\n'''\n" + prompt + "\n'''"
        )

        payload = {"prompt": body_prompt, "max_tokens": 64, "temperature": 0.0}
        if self.model:
            payload["model"] = self.model

        for ep in endpoints:
            try:
                resp = requests.post(ep, json=payload, headers=headers, timeout=self.timeout)
                if not resp.ok:
                    continue
                # try JSON parse first
                try:
                    j = resp.json()
                    # common shapes: choices[0].text | outputs/output/result
                    if isinstance(j, dict):
                        if "choices" in j and j["choices"]:
                            first = j["choices"][0]
                            if isinstance(first, dict):
                                text = first.get("text") or first.get("message", {}).get("content")
                                if text:
                                    return str(text).strip()
                        for k in ("output", "outputs", "result"):
                            if k in j:
                                val = j[k]
                                if isinstance(val, str):
                                    return val.strip()
                                if isinstance(val, list) and val:
                                    return str(val[0]).strip()
                    # fallback to stringified JSON
                    return str(j).strip()
                except Exception:
                    # fallback to raw text body
                    return resp.text.strip()
            except Exception:
                continue

        raise RuntimeError(f"Failed to reach vllm at {self.vllm_url}")

    def _parse_vllm_jsonish(self, out: str) -> Tuple[str, float]:
        """
        Extract (language, confidence) from textual model output.
        Returns ('', 0.0) if parsing fails.
        """
        if not out:
            return "", 0.0

        # JSON-like extraction
        lang_m = re.search(r'"language"\s*:\s*"([^"]+)"', out, re.IGNORECASE)
        conf_m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', out, re.IGNORECASE)
        if lang_m:
            lang = lang_m.group(1).strip().lower()
            conf = float(conf_m.group(1)) if conf_m else 1.0
            return lang, max(0.0, min(1.0, conf))

        # simple patterns: "Italian (0.95)" or "it:0.9"
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

        # keyword heuristic
        if re.search(r'\bital(?:ian|iano)?\b', out, re.IGNORECASE):
            return "it", 1.0

        return "", 0.0

    def _is_italian_score(self, text: str) -> float:
        """
        Override: ask vllm for language + confidence; fallback to parent on errors.
        """
        txt = (text or "").strip()
        if not txt:
            return 0.0

        try:
            out = self._call_vllm(txt[:2000])
            lang, conf = self._parse_vllm_jsonish(out)
            if not lang:
                return super()._is_italian_score(text)
            if lang in ("it", "ita", "italian", "italiano"):
                return float(conf)
            return 0.0
        except Exception:
            return super()._is_italian_score(text)
