from pathlib import Path
import PyPDF2

from archaeo_super_prompt.debug_log import print_log

def get_all_samples_files(dir: Path):
    if not dir.is_dir():
        raise Exception("the given path is not a directory")
    return (p for p in dir.iterdir() if p.is_file() and p.name.endswith(".pdf"))

def pdf_to_text(pdf_path_str: Path) -> str:
    """Extract raw text from a PDF (simple text-only approach)."""
    path = Path(pdf_path_str).resolve()
    text_parts = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def save_log_in_file(fp: str, content: str):
    path = Path(fp).resolve()
    with path.open("w") as f:
        f.write(content)
    print_log(f"Content saved in {fp}\n")
