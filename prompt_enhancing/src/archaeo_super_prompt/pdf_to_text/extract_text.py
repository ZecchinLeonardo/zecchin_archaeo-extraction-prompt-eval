from typing import List
import pdftotext
from pathlib import Path

from ..signatures.input import ExtractedPDFContent
from ..cache import memory


@memory.cache
def extract_text_from_pdf(pdf_path: Path) -> ExtractedPDFContent:
    with pdf_path.open("rb") as pdf_file:
        pages: List[str] = [p for p in pdftotext.PDF(pdf_file)]
        if len(pages) == 0:
            raise Exception("Cannot extract text from this pdf")
        if len(pages) <= 2:
            return { "incipit": pages, "body": pages }
        return { "incipit": pages[:2], "body": pages[2:] }
