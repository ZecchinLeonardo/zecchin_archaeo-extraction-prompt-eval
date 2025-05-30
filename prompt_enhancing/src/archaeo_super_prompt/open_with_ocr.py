import os
import re
from pathlib import Path
from typing import List, Optional, cast
import pymupdf
import pymupdf4llm

from archaeo_super_prompt.debug_log import print_debug_log, print_log

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
    # TODO: reput this correct version
    # return (p for p in dir.iterdir() if p.is_file() and p.name.endswith(".pdf"))
    return (p for p in dir.iterdir() if p.is_file() and p.name.endswith(".txt"))

def normalize_alpha_words(text):
        # Keep only letters and spaces
        cleaned = re.sub(r'[^A-Za-z\s]', '', text)
        # Normalize multiple spaces to one
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Trim leading/trailing spaces
        return cleaned.strip()

def does_the_content_contains_text(content: str) -> bool:
    return len(normalize_alpha_words(content)) > 500

def pdf_to_text(pdf_path: Path) -> str:
    with pdf_path.resolve().open("r") as f:
        return f.read()
    # TODO: reput this more correct (but not completely) code
    # """Extract raw text from a PDF (simple text-only approach)."""
    # path = Path(pdf_path).resolve()

    # print_debug_log(f"Processing document {path}")
    # pdf_document = pymupdf.open(path)
    # markdownified_pdf = pymupdf4llm.to_markdown(pdf_document)
    # if does_the_content_contains_text(markdownified_pdf):
    #     print_debug_log("The document has enough encoded text content.")
    #     return markdownified_pdf

    # # TODO: OCR is not working
    # # using ocr for scanning the document
    # pdf_document_with_text = pymupdf.open()
    # pno = 0
    # last_ref_text_page: Optional[pymupdf.TextPage] = None # use the last pattern identified after an OCR run
    #                                                       # if nothing is scanned, rerun the OCR then memorize
    #                                                       # the last pattern
    # for page in pdf_document:
    #     # page_extracted_text_within_ocr: str = (
    #     #     page.get_text(textpage=last_ref_text_page) # type: ignore
    #     #     if last_ref_text_page is not None
    #     #     else ""
    #     # )  
    #     page_extracted_text_within_ocr: str = ""
    #     if not does_the_content_contains_text(page_extracted_text_within_ocr):
    #         last_ref_text_page = cast(pymupdf.TextPage, page.get_textpage_ocr(language='ita')) #type: ignore
    #         page_extracted_text_within_ocr = page.get_text(textpage=last_ref_text_page) #type: ignore

    #     pdf_document_with_text.insert_page(pno-1, text=page_extracted_text_within_ocr) #type: ignore
    #     pno += 1
    # pdf_document_with_text.save(Path(f"./outputs/{pdf_path.name}.ocr.pdf"))
    # return pymupdf4llm.to_markdown(pdf_document_with_text)

def save_log_in_file(fp: str, content: str):
    path = Path(fp).resolve()
    with path.open("w") as f:
        f.write(content)
    print_log(f"Content saved in {fp}\n")
