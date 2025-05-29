import os
from pathlib import Path
from typing import List
import pymupdf

from archaeo_super_prompt.debug_log import print_log

def init_ocr_setup():
    if "TESSDATA_PREFIX" in os.environ:
        return
    prefix = os.getenv("TESSDATA_PREFIX_DIR")
    if (prefix is None):
        raise EnvironmentError("TESSDATA_PREFIX_DIR is not defined in the .env")
    os.environ["TESSDATA_PREFIX"] = str(Path(prefix).resolve())

def get_all_samples_files(dir: Path):
    if not dir.is_dir():
        raise Exception("the given path is not a directory")
    return (p for p in dir.iterdir() if p.is_file() and p.name.endswith(".pdf"))

def pdf_to_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF (simple text-only approach)."""
    path = Path(pdf_path).resolve()
    text_parts: List[str] = []
    with pymupdf.open(path) as pdf_document:
        for page in pdf_document:
            tp = page.get_textpage_ocr() #type: ignore
            text: str = page.get_text(textpage=tp) #type: ignore
            text_parts.append(text)
    # TODO: convert into markdown with pymupdf4llm
    return "\n".join(text_parts)

def save_log_in_file(fp: str, content: str):
    path = Path(fp).resolve()
    with path.open("w") as f:
        f.write(content)
    print_log(f"Content saved in {fp}\n")
