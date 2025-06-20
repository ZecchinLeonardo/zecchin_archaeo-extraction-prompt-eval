from functools import reduce
import subprocess
from pathlib import Path
from typing import List
from tqdm import tqdm
import threading
import time
import queue

from archaeo_super_prompt.cache import get_cache_subdir

OUTPUT_DIR = get_cache_subdir("ocr-pdfs").resolve()
EXPECTED_COMMON_INPUT_DIRECTORY = get_cache_subdir("pdfs").resolve()


def _get_output_pdfs(input_files: List[Path]):
    return map(lambda p: OUTPUT_DIR / Path(p.parent.name) / p.name, input_files)


def _add_ocr_layer(input_files: List[Path]):
    # Normally, "./cache/pdfs"
    common_input_directory = input_files[0].parent.parent.resolve()
    assert common_input_directory == EXPECTED_COMMON_INPUT_DIRECTORY
    output_pdfs = list(_get_output_pdfs(input_files))
    missing_files = list(
        filter(
            lambda path: not path.exists(),
            output_pdfs,
        )
    )
    for missing_file in missing_files:
        missing_file.parent.mkdir(parents=True, exist_ok=True)

    if missing_files:
        input_file_stems = map(
            lambda path: str(Path(path.parent.name) / path.name), missing_files
        )
        process = subprocess.Popen(
            [
                "just",
                "ocr",
                common_input_directory,
                OUTPUT_DIR,
                *input_file_stems,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            cwd=str(Path(__file__).parent),
        )
        for line in iter(process.stdout.readline, b""):  # type: ignore
            print(f"\n[OCRMYPDF] {line.decode().rstrip()}")

    return output_pdfs


def _inspect_file_number():
    def pdf_nb_in_dir(dir: Path):
        if not dir.is_dir():
            return 0
        return len([True for file in dir.iterdir() if file.suffix == ".pdf"])

    return reduce(lambda sum_, dir: sum_ + pdf_nb_in_dir(dir), OUTPUT_DIR.iterdir(), 0)


def _log_progress(
    current_file_nb: int,
    already_scanned_file_nb: int,
    file_to_produce_nb: int,
    stop_event: threading.Event,
):
    progress_bar = tqdm(
        desc="Produced PDF/A files",
        total=file_to_produce_nb,
        initial=already_scanned_file_nb,
    )
    while not stop_event.is_set():
        added_file_nb = _inspect_file_number() - current_file_nb
        if added_file_nb > 0:
            current_file_nb += added_file_nb
            progress_bar.update(added_file_nb)
        time.sleep(1)
    progress_bar.close()


def _batch_add_ocr_layer(input_files: List[Path], result_pipe: queue.Queue):
    batch_size = 25
    batches = [
        input_files[i : i + batch_size] for i in range(0, len(input_files), batch_size)
    ]
    output_dirs = []
    for batch in batches:
        output_dirs += _add_ocr_layer(batch)
    result_pipe.put(output_dirs)


def add_ocr_layer(input_files: List[Path]):
    file_to_produce_nb = len(input_files)
    already_scanned_file_nb = len(
        list(filter(lambda path: path.exists(), _get_output_pdfs(input_files)))
    )
    current_file_nb = _inspect_file_number()

    stop_event = threading.Event()
    pbar_thread = threading.Thread(
        target=_log_progress,
        args=(current_file_nb, already_scanned_file_nb, file_to_produce_nb, stop_event),
    )
    result_pipe: queue.Queue[List[Path]] = queue.Queue(1)
    ocr_thread = threading.Thread(
        target=_batch_add_ocr_layer, args=(input_files, result_pipe)
    )

    pbar_thread.start()
    ocr_thread.start()
    ocr_thread.join()
    stop_event.set()
    return result_pipe.get()
