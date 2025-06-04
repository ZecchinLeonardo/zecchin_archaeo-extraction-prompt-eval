import re
from pathlib import Path

def get_report_input(filepath: Path) -> str:
    with filepath.open("r") as f:
        return f.read()

def get_all_samples_files(dir: Path):
    """Get all the extracted-text files that has been to be processed in
    the pipeline
    """
    if not dir.is_dir():
        raise Exception(f"The given path is not a directory: {dir}")
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
