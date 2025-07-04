"""Better OCR model with VLLM"""

from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple
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


def _stream_document_pages(file: Path) -> Tuple[Iterable[DocumentStream], int]:
    source_doc = pymupdf.open(file)

    def create_smaller_pdf_for_page(source_doc: pymupdf.Document, page_number: int):
        per_page_pdf = pymupdf.open()
        per_page_pdf.insert_pdf(source_doc, from_page=page_number, to_page=page_number)
        return DocumentStream(
            name=f"{source_doc.name}__p{page_number}.pdf",
            stream=BytesIO(per_page_pdf.tobytes()),
        )

    pages = (
        create_smaller_pdf_for_page(source_doc, pn)
        for pn in range(source_doc.page_count)
    )
    return pages, source_doc.page_count


def ollama_vlm_options(model: str, prompt: str, response_format: ResponseFormat):
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
        timeout=60,
        concurrency=2,
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


def process_documents(files: List[Path], documentConvertor: DocumentConverter):
    results_over_files: List[List[ConversionResult]] = []
    for file in tqdm(files, desc="processed files"):
        page_stream, page_number = _stream_document_pages(file)
        results = documentConvertor.convert_all(page_stream)
        results_over_files.append([result for result in tqdm(results, desc="VLLM processed pages", total=page_number, unit="page")])
    return results_over_files


def process_documents__ollma_ocr(files: List[Path]):
    print(str(files[0]))
    results = _ocr.process_batch(
        [str(f) for f in files],
        format_type="structured",
        language="ita",
    )
    return results
