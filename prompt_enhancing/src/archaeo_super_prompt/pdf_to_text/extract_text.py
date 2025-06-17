import pdftotext
from pathlib import Path
from ..cache import memory

@memory.cache
def extract_text_from_pdf(pdf_path: Path) -> str:
    with pdf_path.open("rb") as pdf_file:
        return "\n\n".join(pdftotext.PDF(pdf_file))
