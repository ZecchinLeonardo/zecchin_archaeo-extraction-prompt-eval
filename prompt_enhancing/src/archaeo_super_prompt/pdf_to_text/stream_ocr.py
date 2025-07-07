"""Better OCR model with VLLM"""

from io import BytesIO
from pathlib import Path
from typing import Generator, Iterable, List, Literal, Tuple
from docling.datamodel.document import ConversionResult
from ollama_ocr import OCRProcessor
from pydantic import AnyUrl
import pymupdf
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.base_models import DocumentStream


_ocr = OCRProcessor(model_name="granite3.2-vision")
_PARALLEL_PAGE_NB = 2


def _document_page_number(file: Path) -> int:
    source_doc = pymupdf.open(file)
    page_count = source_doc.page_count
    source_doc.close()
    return page_count


def stream_document_pages(file: Path) -> Tuple[Iterable[DocumentStream], int]:
    """We do not use a per-page processing to let Docling running its native
    per-page vllm processing, hoping more performant results will be output.

    So this function is kept here in case of
    """
    source_doc = pymupdf.open(file)

    def create_smaller_pdf_for_page(page_number: int):
        per_page_pdf = pymupdf.open()
        per_page_pdf.insert_pdf(source_doc, from_page=page_number, to_page=page_number)
        return DocumentStream(
            # intervention_id + fileanme + page_number + .pdf
            name=f"{file.parent.name}__{file.stem}__p{page_number}.pdf",
            stream=BytesIO(per_page_pdf.tobytes()),
        )

    pages = (create_smaller_pdf_for_page(pn) for pn in range(source_doc.page_count))
    return pages, source_doc.page_count


def ollama_vlm_options(
    model: str,
    prompt: str,
    response_format: Literal[
        ResponseFormat.HTML, ResponseFormat.MARKDOWN
    ] = ResponseFormat.MARKDOWN,
    allowed_timeout: int = 60 * 3,
):
    """Arguments:
    * model: the string identifier of the vllm model in ollama
    * prompt: a string to prompt to the vllm to contextualize its OCR task
    * response_format: a supported response format for the vllm
    * allowed_timeout: the allowed time for processing one page in one
    document (default to 3 minutes)
    """
    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.
    #
    # One possibility is self-hosting model, e.g. via LM Studio, Ollama or others.
    options = ApiVlmOptions(
        url=AnyUrl(
            "http://localhost:11434/v1/chat/completions"
        ),  # the default Ollama endpoint
        params=dict(
            model=model,
        ),
        prompt=prompt,
        # One page may take 3 minutes to be roughly well processed
        timeout=allowed_timeout,
        concurrency=_PARALLEL_PAGE_NB,
        scale=1.0,
        response_format=response_format,
    )
    return options


def converter(ollama_vlm_options: ApiVlmOptions):
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # <-- this is required!
    )

    # Example using the Granite Vision model with Ollama:
    pipeline_options.vlm_options = ollama_vlm_options
    # Create the DocumentConverter and launch the conversion.
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )
    return doc_converter


def process_documents(
    files: List[Path], documentConvertor: DocumentConverter, timeout_per_page: int
):
    results_over_files: List[ConversionResult] = []
    page_counts = (_document_page_number(p) for p in files)
    result_iterable = documentConvertor.convert_all(
        files, max_num_pages=150, raises_on_error=False
    )
    print(
        "Max duration for the current scan:", next(page_counts) * timeout_per_page, "s"
    )
    for result in tqdm(result_iterable, desc="vllm-scanned files", unit="file"):
        try:
            print(
                "Max duration for the current scan:",
                next(page_counts) * timeout_per_page,
                "s",
            )
        except StopIteration:
            # edge-case for the last loop
            pass
        results_over_files.append(result)

    return results_over_files


def process_documents__ollma_ocr(files: List[Path]):
    print(str(files[0]))
    results = _ocr.process_batch(
        [str(f) for f in files],
        format_type="structured",
        language="ita",
    )
    return results
