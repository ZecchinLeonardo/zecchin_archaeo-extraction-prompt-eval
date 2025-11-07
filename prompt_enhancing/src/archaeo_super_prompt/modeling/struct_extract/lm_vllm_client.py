"""
Lightweight vllm HTTP client + token-logprob -> confidence helpers.

Usage:
  from archaeo_super_prompt.modeling.struct_extract.lm_vllm_client import (
      call_vllm_generate, compute_field_confidences
  )

  resp = call_vllm_generate(prompt, max_tokens=256, temperature=0.0)
  # resp contains 'text', 'tokens', 'token_logprobs', 'raw'
  overall_conf, per_field_conf = compute_field_confidences(
      resp["text"], resp["tokens"], resp["token_logprobs"], parsed_fields
  )

Place this file under:
prompt_enhancing/src/archaeo_super_prompt/modeling/struct_extract/
"""
from __future__ import annotations
import requests
import math
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Configure the VLLM server base URL here or override at runtime
DEFAULT_VLLM_URL = "http://localhost:8001"


def call_vllm_generate(
    prompt: str,
    vllm_url: str = DEFAULT_VLLM_URL,
    max_tokens: int = 256,
    temperature: float = 0.0,
    logprobs: bool = True,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Call the vllm server and request token logprobs.

    Returns a normalized dictionary with at least:
      - text: full generated text
      - tokens: list[str] of token strings (in generation order)
      - token_logprobs: list[float] aligned with tokens (natural log)
      - raw: original JSON response

    The function tries common endpoints and common response shapes.
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # vllm compatibility: some servers accept "logprobs": True, some accept integer.
    if logprobs:
        payload["logprobs"] = True

    # endpoints to try (order matters)
    try_paths = ["/v1/generate", "/v1/completions", "/v1/chat/completions", "/v1/completions/stream"]
    last_exc: Optional[Exception] = None
    for path in try_paths:
        url = vllm_url.rstrip("/") + path
        try:
            r = requests.post(url, json=payload, timeout=timeout)
        except Exception as e:
            last_exc = e
            continue
        if not r.ok:
            last_exc = RuntimeError(f"HTTP {r.status_code}: {r.text}")
            continue
        resp = r.json()
        # Normalize common shapes
        choice = None
        if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
            choice = resp["choices"][0]
        elif "outputs" in resp and isinstance(resp["outputs"], list) and resp["outputs"]:
            choice = resp["outputs"][0]
        else:
            # unknown shape; return raw
            return {"text": "", "tokens": [], "token_logprobs": [], "raw": resp}

        # possible text fields
        text = ""
        if "text" in choice and isinstance(choice["text"], str):
            text = choice["text"]
        elif "message" in choice and isinstance(choice["message"], dict):
            # chat shape
            text = choice["message"].get("content", "")
        elif "content" in choice:
            text = choice.get("content", "")
        else:
            # fallback to concatenating token texts if available
            lp = choice.get("logprobs") or {}
            tokens = lp.get("tokens") or []
            text = "".join(tokens) if tokens else ""

        # attempt to extract tokens + token_logprobs
        tokens = []
        token_logprobs: List[float] = []
        # common shape: choice["logprobs"]["tokens"], choice["logprobs"]["token_logprobs"]
        logp_block = choice.get("logprobs") or {}
        if isinstance(logp_block, dict):
            tokens = logp_block.get("tokens") or logp_block.get("token_strs") or []
            token_logprobs = logp_block.get("token_logprobs") or logp_block.get("token_logprob") or []
        # some servers embed token_logprobs directly on choice
        if not tokens and isinstance(choice.get("tokens"), list):
            tokens = choice.get("tokens")
        if not token_logprobs and isinstance(choice.get("token_logprobs"), list):
            token_logprobs = choice.get("token_logprobs")

        # normalize lengths: convert None to 0.0 and ensure same length
        if tokens and token_logprobs and len(tokens) != len(token_logprobs):
            # if lengths mismatch, prefer tokens and pad token_logprobs with None
            if len(token_logprobs) < len(tokens):
                token_logprobs = token_logprobs + [None] * (len(tokens) - len(token_logprobs))
            else:
                token_logprobs = token_logprobs[: len(tokens)]

        # finally, if tokens is empty: try to reconstruct from text by splitting on spaces (fallback)
        if not tokens:
            # very coarse fallback: split by characters to keep alignment impossible but non-empty
            tokens = [text] if text else []

        # convert token_logprobs to floats (None stays None)
        token_logprobs = [float(x) if x is not None else None for x in token_logprobs] if token_logprobs else []

        return {"text": text, "tokens": tokens, "token_logprobs": token_logprobs, "raw": resp}

    # If we get here no endpoint worked
    raise last_exc or RuntimeError("No vllm endpoint reachable")


# -----------------------
# token <-> char alignment
# -----------------------
def tokens_to_char_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Build char-level spans for tokens when concatenated in order.
    This assumes tokens are returned with their whitespace preserved (vllm often does).
    """
    spans: List[Tuple[int, int]] = []
    s = ""
    for t in tokens:
        start = len(s)
        s += t
        end = len(s)
        spans.append((start, end))
    return spans


def find_token_indices_for_substring(tokens: List[str], substring: str) -> Optional[Tuple[int, int]]:
    """
    Find a contiguous token index span that covers an instance of `substring` inside the reconstruction
    of the tokens. Returns (start_idx, end_idx) inclusive, or None if not found.
    """
    if not substring:
        return None
    reconstructed = "".join(tokens)
    idx = reconstructed.find(substring)
    if idx == -1:
        return None
    spans = tokens_to_char_spans(tokens)
    # first token with span end > idx
    start_idx = next((i for i, (_a, b) in enumerate(spans) if b > idx), 0)
    end_pos = idx + len(substring)
    end_idx = len(tokens) - 1
    for i, (a, b) in enumerate(spans):
        if a >= end_pos:
            end_idx = i - 1
            break
    # sanity clamp
    start_idx = max(0, min(start_idx, len(tokens) - 1))
    end_idx = max(start_idx, min(end_idx, len(tokens) - 1))
    return (start_idx, end_idx)


# -----------------------
# aggregation helpers
# -----------------------
def geo_mean_prob_from_logprobs(token_logprobs: List[Optional[float]]) -> float:
    """
    Given token logprobs (natural log), compute geometric-mean probability:
      exp(mean(logp)) -> in (0,1)
    If token_logprobs contains None, those tokens are ignored from aggregation.
    """
    valid = [float(lp) for lp in token_logprobs if lp is not None]
    if not valid:
        return 0.0
    avg_logp = mean(valid)
    # clamp numeric overflow/underflow edgecases
    val = math.exp(avg_logp)
    if not math.isfinite(val):
        # if avg_logp is very small negative, exp may underflow -> return 0.0
        return 0.0
    return max(0.0, min(1.0, val))


def compute_field_confidences(
    full_text: str,
    tokens: List[str],
    token_logprobs: List[Optional[float]],
    parsed_fields: Dict[str, Any],
    fallback_to_global: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-field confidences using token logprobs aligned to field substrings.

    Arguments:
      parsed_fields: mapping field_name -> predicted string value (the value you parsed from output)
    Returns:
      (overall_confidence, {field: conf})
      overall_confidence is the mean of per-field confidences (0..1).
    """
    field_confidences: Dict[str, float] = {}

    reconstructed = "".join(tokens) if tokens else full_text

    for field, value in parsed_fields.items():
        sval = "" if value is None else str(value).strip()
        if not sval:
            field_confidences[field] = 0.0
            continue
        # try token alignment
        span = find_token_indices_for_substring(tokens, sval)
        if span is None:
            # fallback behaviour
            if fallback_to_global:
                # use global token_logprobs
                conf = geo_mean_prob_from_logprobs(token_logprobs)
            else:
                conf = 0.0
        else:
            a, b = span
            slice_logps = token_logprobs[a : b + 1]
            conf = geo_mean_prob_from_logprobs(slice_logps)
        field_confidences[field] = conf

    if field_confidences:
        overall = sum(field_confidences.values()) / len(field_confidences)
    else:
        overall = 0.0
    return float(overall), {k: float(v) for k, v in field_confidences.items()}