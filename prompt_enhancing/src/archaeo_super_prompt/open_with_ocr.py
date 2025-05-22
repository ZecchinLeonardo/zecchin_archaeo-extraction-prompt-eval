from pathlib import Path
import PyPDF2

def pdf_to_text(pdf_path_str: str) -> str:
    """Extract raw text from a PDF (simple text-only approach)."""
    path = Path(pdf_path_str).resolve()
    text_parts = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)
