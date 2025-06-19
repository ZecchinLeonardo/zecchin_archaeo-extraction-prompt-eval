import subprocess
import sys
from pathlib import Path
from typing import List

OUTPUT_DIR = Path("./.cache/ocr-pdfs/").resolve()


def add_ocr_layer(input_files: List[Path]):
    # Normally, "./cache/pdfs"
    common_input_directory = input_files[0].parent.parent.resolve()
    assert common_input_directory == Path("./.cache/pdfs").resolve()
    file_stems = [Path(p.parent.name) / p.name for p in input_files]
    missing_files = list(
        filter(
            lambda path: not (OUTPUT_DIR / path).exists(),
            file_stems,
        )
    )
    for missing_file in missing_files:
        (OUTPUT_DIR / missing_file.parent).mkdir(parents=True, exist_ok=True)

    if missing_files:
        result = subprocess.run(
            [
                "just",
                "ocr",
                common_input_directory,
                OUTPUT_DIR,
                *[str(f) for f in missing_files],
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        print("Out:", result.stdout)
        print("Error:", result.stderr)
    return [OUTPUT_DIR / file_stem for file_stem in file_stems]
