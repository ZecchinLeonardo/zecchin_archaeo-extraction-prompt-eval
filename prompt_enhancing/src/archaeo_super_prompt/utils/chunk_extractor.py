import re
import ast
from typing import Optional, List, Dict
from rapidfuzz import fuzz



def _extract_chunks_from_merged(merged: str) -> list[str]:
    if not merged:
        return []
    sep = "`" + "-" * 60 + "`\n\n"
    blocks = [b.strip() for b in merged.split(sep) if b.strip()]
    # each block currently is like: "`%% filename | Page N (type) %%`\n\n<chunk_content>"
    chunks = []
    for b in blocks:
        parts = b.split("\n\n", 1)
        if len(parts) == 2:
            chunks.append(parts[1].strip())
        else:
            # fallback: take entire block
            chunks.append(b.strip())
    return chunks


def _limit_context_sentences(chunk: str, name: str, window: int = 2) -> str:
    if not chunk:
        return ""
    # split into sentences (keeps punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    if not sentences:
        return chunk.strip()
    lname = (name or "").lower().strip()

    # try to find sentence containing the name
    found_idx = None
    if lname:
        for i, s in enumerate(sentences):
            if lname in s.lower():
                found_idx = i
                break

    # if not found, pick best-matching sentence by fuzzy score
    if found_idx is None:
        best_idx = 0
        best_score = -1
        for i, s in enumerate(sentences):
            score = fuzz.token_sort_ratio(lname, s.lower())
            if score > best_score:
                best_score = score
                best_idx = i
        found_idx = best_idx

    start = max(0, found_idx - window)
    end = min(len(sentences) - 1, found_idx + window)
    return " ".join(sentences[start : end + 1]).strip()


def _choose_best_chunk_for_name(merged: str, name: str) -> str:
    if not merged:
        return ""
    chunks = _extract_chunks_from_merged(merged)
    if not chunks:
        return ""
    if not name:
        return ""  # unknown name -> nothing
    lname = name.lower().strip()

    # Try exact substring search first (return limited context)
    for c in chunks:
        if lname in c.lower():
            return _limit_context_sentences(c, name, window=2)

    # Fallback: fuzzy match on sentences within chunks
    best_chunk = ""
    best_score = -1.0
    for c in chunks:
        # compute chunk-level score as best sentence match inside it
        sentences = re.split(r'(?<=[.!?])\s+', c)
        for s in sentences:
            score = fuzz.token_sort_ratio(lname, s.lower())
            if score > best_score:
                best_score = score
                best_chunk = c

    # require a minimal confidence to return a context, otherwise empty
    if best_score >= 40:
        return _limit_context_sentences(best_chunk, name, window=2)
    return ""

# def _extract_chunks_info_from_merged(merged: str) -> list[Dict]:
#     """Return list of {header, content, filename, page, chunk_type} for each block."""
#     if not merged:
#         return []
#     sep = "`" + "-" * 60 + "`\n\n"
#     blocks = [b.strip() for b in merged.split(sep) if b.strip()]
#     infos: list[Dict] = []
#     header_re = re.compile(r'`?%+ ?(?P<filename>.*?) \| Page (?P<page>\d+)(?: \((?P<type>.*?)\))? ?%+`?')
#     for b in blocks:
#         # header + blank line + content
#         parts = b.split("\n\n", 1)
#         header = parts[0].strip() if parts else ""
#         content = parts[1].strip() if len(parts) == 2 else ""
#         m = header_re.search(header)
#         filename = m.group("filename").strip() if m else ""
#         page = int(m.group("page")) if m and m.group("page") and m.group("page").isdigit() else None
#         ctype = m.group("type").strip() if m and m.group("type") else ""
#         infos.append({"header": header, "content": content, "filename": filename, "page": page, "chunk_type": ctype})
#     return infos

# def _choose_best_chunk_and_page(merged: str, name: str, window: int = 2) -> tuple[str, Optional[int]]:
#     """Return (limited_context, page) for best matching chunk; page may be None."""
#     if not merged:
#         return "", None
#     infos = _extract_chunks_info_from_merged(merged)
#     if not infos:
#         return "", None
#     lname = (name or "").lower().strip()
#     # exact substring candidate
#     for info in infos:
#         if lname and lname in info["content"].lower():
#             return _limit_context_sentences(info["content"], name, window=window), info["page"]
#     # fuzzy sentence matching
#     best_info = None
#     best_score = -1
#     for info in infos:
#         sentences = re.split(r'(?<=[.!?])\s+', info["content"])
#         for s in sentences:
#             score = fuzz.token_sort_ratio(lname, s.lower()) if lname else 0
#             if score > best_score:
#                 best_score = score
#                 best_info = info
#     if best_info and best_score >= 40:
#         return _limit_context_sentences(best_info["content"], name, window=window), best_info["page"]
#     return "", None

def _extract_chunks_info_from_merged(merged: str) -> list[Dict]:
    """Return list of {header, content, filename, page, chunk_type} for each block.
    Handles headers like:
      upload_f07e0c63.pdf | Page [1] (['text'])
      `%% upload_f07e0c63.pdf | Page 1 (text) %%`
    """
    if not merged:
        return []
    sep = "`" + "-" * 60 + "`\n\n"
    blocks = [b.strip() for b in merged.split(sep) if b.strip()]
    infos: list[Dict] = []
    # permissive header regex (accepts optional square brackets around page)
    header_re = re.compile(
        r"(?P<filename>.*?)\s*\|\s*Page\s*\[?(?P<page>\d+)\]?\s*(?:\((?P<type>.*)\))?",
        flags=re.IGNORECASE,
    )
    for b in blocks:
        parts = b.split("\n\n", 1)
        header = parts[0].strip() if parts else ""
        content = parts[1].strip() if len(parts) == 2 else ""
        filename = ""
        page = None
        ctype = ""
        m = header_re.search(header)
        if m:
            filename = m.group("filename").strip()
            page = int(m.group("page")) if m.group("page") and m.group("page").isdigit() else None
            raw_type = m.group("type")
            if raw_type:
                # try to parse Python-list-like strings such as "['text']"
                try:
                    parsed = ast.literal_eval(raw_type)
                    if isinstance(parsed, (list, tuple)) and parsed:
                        ctype = str(parsed[0])
                    else:
                        ctype = str(parsed)
                except Exception:
                    # fallback: strip wrappers and quotes
                    ctype = re.sub(r"^[\[\('\" ]+|[\]\)'\"]+$", "", raw_type).strip()
        else:
            # fallback: find Page <num> anywhere
            mm = re.search(r"Page\s*\[?(\d+)\]?", header, flags=re.IGNORECASE)
            if mm:
                page = int(mm.group(1))
            fn_match = re.match(r"[`%]*\s*(?P<filename>[^|`%]+)", header)
            if fn_match:
                filename = fn_match.group("filename").strip()
        infos.append({"header": header, "content": content, "filename": filename, "page": page, "chunk_type": ctype})
    return infos


def _choose_best_chunk_and_page(merged: str, name: str, window: int = 2) -> tuple[str, Optional[int]]:
    """Return (limited_context, page) for best matching chunk; page may be None."""
    if not merged:
        return "", None
    infos = _extract_chunks_info_from_merged(merged)
    if not infos:
        return "", None
    lname = (name or "").lower().strip()

    # exact substring candidate (prefer exact matches)
    if lname:
        for info in infos:
            if lname in info["content"].lower():
                return _limit_context_sentences(info["content"], name, window=window), info["page"]

    # fallback: fuzzy sentence matching inside chunks
    best_info = None
    best_score = -1
    for info in infos:
        sentences = re.split(r'(?<=[.!?])\s+', info["content"])
        for s in sentences:
            score = fuzz.token_sort_ratio(lname, s.lower()) if lname else 0
            if score > best_score:
                best_score = score
                best_info = info

    if best_info and best_score >= 40:
        return _limit_context_sentences(best_info["content"], name, window=window), best_info["page"]

    # final fallback: return first chunk's limited context and its page
    first = infos[0]
    return _limit_context_sentences(first["content"], name, window=window), first["page"]