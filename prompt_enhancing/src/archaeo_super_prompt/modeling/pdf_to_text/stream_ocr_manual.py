"""Better OCR model with VLLM."""

from pathlib import Path
from typing import (
    cast,
    Literal,
)
from collections.abc import Iterator
from pydantic import AnyUrl
import pymupdf
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import PageRange
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from .types import has_document_been_well_scanned, CorrectlyConvertedDocument
from ...types.intervention_id import InterventionId
from .document_division import get_page_ranges

from ...config.debug_log import print_log
from ...config.env import getenv_or_throw
from ...utils import cache

from . import cache_docling_documents as cache_dd


_PARALLEL_PAGE_NB = 2

INCIPIT_MAX_PAGES = 5


def _document_page_number(file: Path) -> int:
    source_doc = pymupdf.open(file)
    page_count = source_doc.page_count
    source_doc.close()
    return page_count


def ollama_vlm_options(
    model: str,
    prompt: str,
    response_format: Literal[
        ResponseFormat.HTML, ResponseFormat.MARKDOWN
    ] = ResponseFormat.MARKDOWN,
    allowed_timeout: int = 60 * 3,
):
    """Return a configuration for vlm model set with ollama.

    Arguments:
        model: the string identifier of the vllm model in ollama
        prompt: a string to prompt to the vllm to contextualize its OCR task
        response_format: a supported response format for the vllm
        allowed_timeout: the allowed time for processing one page in one \
document (default to 3 minutes)
    """
    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.
    #
    # One possibility is self-hosting model, e.g. via LM Studio, Ollama or others.
    options = ApiVlmOptions(
        url=AnyUrl(
            f"{getenv_or_throw("VLM_HOST_URL")}/v1/chat/completions"
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


def vllm_vlm_options(
    model: str,
    prompt: str,
    response_format: Literal[
        ResponseFormat.HTML, ResponseFormat.MARKDOWN
    ] = ResponseFormat.MARKDOWN,
    allowed_timeout: int = 60 * 3,
):
    """Return a configuration for vlm model set with a vllm server (so an OpenAI compatible API).

    Arguments:
        model: the string identifier of the vllm model in ollama
        prompt: a string to prompt to the vllm to contextualize its OCR task
        response_format: a supported response format for the vllm
        allowed_timeout: the allowed time for processing one page in one \
document (default to 3 minutes)
    """
    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.
    #
    # One possibility is self-hosting model, e.g. via LM Studio, Ollama or others.
    options = ApiVlmOptions(
        url=AnyUrl(
            f"{getenv_or_throw("VLM_HOST_URL")}/v1/chat/completions"
        ),  # an arbitraty port
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
    """Return a Docling PDF converter object from an ollama vlm configuration."""
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


def _process_page_ranges_with_cache(
    intervention_id: InterventionId,
    file: Path,
    docConverter: DocumentConverter,
    page_ranges: Iterator[PageRange],
) -> Iterator[tuple[PageRange, CorrectlyConvertedDocument | None]]:
    def scan_page_range(
        page_ranges: Iterator[PageRange],
    ) -> Iterator[CorrectlyConvertedDocument | None]:
        return (
            has_document_been_well_scanned(
                docConverter.convert(
                    file,
                    page_range=p_range,
                    raises_on_error=False,
                )
            )
            for p_range in page_ranges
        )

    def get_yaml_file_for_pdf_slice(page_range: PageRange):
        return cache_dd.get_yaml_file_for_pdf(
            cache_dd.ArtificialPDFData(intervention_id, file.stem, page_range)
        )

    return cache.manualy_cache_batch_processing(
        get_yaml_file_for_pdf_slice,
        cache_dd.cache_docling_doc_on_disk,
        cache_dd.load_docling_doc_from_cache,
        scan_page_range,
        page_ranges,
    )


def _retry_scanning_failed_document(
    intervention_id: InterventionId,
    doc: Path,
    docConverter: DocumentConverter,
    page_range: PageRange,
) -> Iterator[tuple[PageRange, CorrectlyConvertedDocument | None]]:
    print_log("Retry scanning the document page per page...")
    page_ranges = [
        cast(PageRange, (p_number, p_number))
        for p_number in range(page_range[0], page_range[1] + 1)
    ]
    return iter(
        tqdm(
            _process_page_ranges_with_cache(
                intervention_id, doc, docConverter, iter(page_ranges)
            ),
            desc=f"({intervention_id}, {page_range[0]}-{page_range[1]}) rescanned pages",
            unit="page",
            total=len(page_ranges),
        )
    )


def _convert_document_with_parallel_pages(
    intervention_id: InterventionId,
    file: Path,
    p_count: int,
    docConverter: DocumentConverter,
    incipit_only: bool,
) -> Iterator[tuple[PageRange, CorrectlyConvertedDocument | None]]:
    page_ranges = get_page_ranges(
        p_count,
        _PARALLEL_PAGE_NB,
        INCIPIT_MAX_PAGES if incipit_only else None,
    )
    return iter(
        tqdm(
            _process_page_ranges_with_cache(
                intervention_id, file, docConverter, iter(page_ranges)
            ),
            desc=f"Doc n°{intervention_id}'s scanned proportion",
            unit="page batch",
            total=len(page_ranges),
        )
    )


def process_documents(
    file_inputs: list[tuple[InterventionId, Path]],
    documentConvertor: DocumentConverter,
    incipit_only=True,
) -> Iterator[
    tuple[
        tuple[InterventionId, Path],
        Iterator[tuple[PageRange, CorrectlyConvertedDocument]],
    ]
]:
    """Convert the documents into text with Docling, using the given converter.

    Return:
    For each file, either a list of one docling document, if all the document
    can have been procesed at once, or a list of nullable docling documents for each
    document page. For some pages, the a null value is put when the page
    reading has failed.
    """

    def convert_all_with_retry(
        intervention_id: InterventionId, file: Path, p_count: int
    ) -> Iterator[tuple[PageRange, CorrectlyConvertedDocument]]:
        for p_range, result in _convert_document_with_parallel_pages(
            intervention_id, file, p_count, documentConvertor, incipit_only
        ):
            if result is not None:
                yield p_range, result
            else:
                for p_range, result in _retry_scanning_failed_document(
                    intervention_id, file, documentConvertor, p_range
                ):
                    if result is not None:
                        yield p_range, result

    return (
        (
            (id_, f),
            convert_all_with_retry(id_, f, p_count),
        )
        for id_, f, p_count in (
            (id_, f, _document_page_number(f))
            for id_, f in tqdm(
                file_inputs,
                desc="vision-llm-scanned files",
                unit="file",
            )
        )
    )
