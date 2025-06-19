import json
import re
from pathlib import Path
from typing import Dict, List

from .signatures.input import ExtractedPDFContent

from .magoh_target import MagohData

ANSWER_JSON_FILENAME = "sample_answers.json"

def _get_report_input(filepath: Path) -> ExtractedPDFContent:
    with filepath.open("r") as f:
        return f.read()

def get_all_samples(dir: Path):
    """Get all the extracted-text files and answer that has been to be processed in
    the pipeline
    """
    if not dir.is_dir():
        raise Exception(f"The given path is not a directory: {dir}")
    answer_fp = dir.joinpath(ANSWER_JSON_FILENAME)
    if not (answer_fp.exists() and answer_fp.is_file()):
        raise Exception(f"{answer_fp} does not exists. Can not load the dataset.")
    samples: List[MagohData]
    with answer_fp.open("r") as answer_file:
        samples = json.load(answer_file)
    for sample in samples:
        id_ = sample["scheda_intervento"]["id"]
        sample_dir = dir.joinpath(str(id_))
        if sample_dir.exists() and sample_dir.is_dir():
            extracted_sources: Dict[str, ExtractedPDFContent] = {
                f.stem: _get_report_input(f)
                for f in sample_dir.iterdir()
                if f.is_file() and f.suffix == ".txt"
            }
            if extracted_sources:
                yield (sample, extracted_sources)

def normalize_alpha_words(text):
    # Keep only letters and spaces
    cleaned = re.sub(r'[^A-Za-z\s]', '', text)
    # Normalize multiple spaces to one
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Trim leading/trailing spaces
    return cleaned.strip()

def does_the_content_contains_text(content: str) -> bool:
    return len(normalize_alpha_words(content)) > 500
