"""Better OCR model with VLLM"""

from pathlib import Path
from typing import List
from ollama_ocr import OCRProcessor
from tqdm import tqdm

_ocr = OCRProcessor(model_name='granite3.2-vision')

def process_documents(files: List[Path]):
    print(str(files[0]))
    results = _ocr.process_batch([str(f) for f in files],
                                format_type="structured",
                                language="ita",
                                )
    return results
