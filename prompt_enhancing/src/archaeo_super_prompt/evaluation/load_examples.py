from pathlib import Path
from typing import List

import dspy

from ..open_with_ocr import get_all_samples

DevSet = List[dspy.Example]

def load_examples(input_file_dir_path: Path) -> DevSet:
    return [
        dspy.Example(
            document_ocr_scan=evaluation_report_dir[1], answer=evaluation_report_dir[0]
        ).with_inputs("document_ocr_scan")
        for evaluation_report_dir in get_all_samples(input_file_dir_path)
    ]
