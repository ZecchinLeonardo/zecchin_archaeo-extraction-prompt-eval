"""
Token-aware chunking helpers.

This module provides:
- estimate_token_count(text): try tiktoken to estimate tokens, fallback to words.
- chunk_text_by_tokens(text, max_tokens): yield text chunks <= max_tokens (using tiktoken if available),
  otherwise fall back to approximate char-based splitting.

Place under:
prompt_enhancing/src/archaeo_super_prompt/modeling/struct_extract/utils/chunking.py
"""
from __future__ import annotations
from typing import Generator, Iterable
import math

# Preferred encoding for many OpenAI-compatible models; works with tiktoken if available.
_DEFAULT_ENCODING = "cl100k_base"


def estimate_token_count(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """
    Try to get an accurate token count using tiktoken. If tiktoken isn't
    available, fall back to a word-count heuristic.
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # rough fallback: assume ~1 token per 3.5-4 characters on average
        return max(1, int(len(text) / 4.0))


def chunk_text_by_tokens(text: str, max_tokens: int, encoding_name: str = _DEFAULT_ENCODING) -> Iterable[str]:
    """
    Yield chunks of `text` where each chunk encodes to at most max_tokens tokens.
    Uses tiktoken if available to split precisely; otherwise uses an approximate
    char-based split (approx 4 chars per token).

    Note: this function yields contiguous slices of the original text (no semantic segmentation).
    For better results you could split on paragraph boundaries inside each token window.
    """
    if not text:
        return []

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
        token_ids = enc.encode(text)
        # slice token ids and decode for each chunk
        for i in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[i : i + max_tokens]
            yield enc.decode(chunk_ids)
    except Exception:
        # fallback: approximate by characters
        # assume ~4 characters per token
        approx_chars = max(1, int(max_tokens * 4))
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + approx_chars)
            # Try to avoid cutting mid-paragraph; extend a bit to the next newline if present
            if end < text_len:
                nl = text.rfind("\n", start, end)
                if nl > start + approx_chars // 4:
                    end = nl
            yield text[start:end]
            start = end