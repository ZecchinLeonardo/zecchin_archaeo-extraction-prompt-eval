"""Better OCR model with VLLM."""

from io import BytesIO
from pathlib import Path
from typing import (
    cast,
    Literal,
)
from collections.abc import Iterator, Iterable
from pydantic import AnyUrl
import pymupdf
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.io import DocumentStream

from .types import has_document_been_well_scanned, CorrectlyConvertedDocument
from ...types.intervention_id import InterventionId
from ...utils import cache

from ...config.debug_log import print_log

from . import cache_docling_documents as cache_dd


_PARALLEL_PAGE_NB = 2


def _document_page_number(file: Path) -> int:
    source_doc = pymupdf.open(file)
    page_count = source_doc.page_count
    source_doc.close()
    return page_count


def stream_document_pages(
    file: Path,
) -> tuple[Iterator[tuple[cache_dd.ArtificialPDFData, DocumentStream]], int]:
    """Load a pdf document and split it into an iterable of buffer for each page.

    We do not use a per-page processing to let Docling running its native
    per-page vllm processing, hoping more performant results will be output.

    So this function is kept here in case of
    """
    source_doc = pymupdf.open(file)

    def create_smaller_pdf_for_page(page_number: int):
        per_page_pdf = pymupdf.open()
        per_page_pdf.insert_pdf(
            source_doc, from_page=page_number, to_page=page_number
        )
        intervention_id = file.parent.name
        filestem = file.stem
        return cache_dd.ArtificialPDFData(
            intervention_id=InterventionId(int(intervention_id)),
            filestem=filestem,
            page_number=page_number,
        ), DocumentStream(
            # intervention_id + fileanme + page_number + .pdf
            name=f"{intervention_id}__{filestem}__p{page_number}.pdf",
            stream=BytesIO(per_page_pdf.tobytes()),
        )

    pages = (
        create_smaller_pdf_for_page(pn) for pn in range(source_doc.page_count)
    )
    return pages, source_doc.page_count


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


def _retry_scanning_failed_document(
    doc: Path, docConverter: DocumentConverter
) -> list[CorrectlyConvertedDocument | None]:
    print_log("Retry scanning the document page per page...")
    pages_iter, page_number = stream_document_pages(doc)
    return [
        page_doc
        for _, page_doc in cache.manualy_cache_batch_processing(
            lambda t: cache_dd.get_yaml_file_for_artificial_pdf(t[0]),
            cache_dd.cache_docling_doc_on_disk,
            cache_dd.load_docling_doc_from_cache,
            lambda tpit: iter(
                tqdm(
                    map(
                        has_document_been_well_scanned,
                        docConverter.convert_all(
                            (pit for _, pit in tpit),
                            max_num_pages=1,
                            raises_on_error=False,
                        ),
                    ),
                    desc="Individual page scan",
                    unit="page",
                    total=page_number,
                )
            ),
            pages_iter,
        )
    ]


def process_documents(
    file_inputs: list[tuple[InterventionId, Path]],
    documentConvertor: DocumentConverter,
    timeout_per_page: int,
) -> list[
    tuple[
        tuple[InterventionId, Path],
        CorrectlyConvertedDocument | list[CorrectlyConvertedDocument | None],
    ]
]:
    """Convert the documents into text with Docling, using the given converter.

    Return:
    For each file, either a list of one docling document, if all the document
    can have been procesed at once, or a list of nullable docling documents for each
    document page. For some pages, the a null value is put when the page
    reading has failed.
    """
    ids, files = cast(
        tuple[list[InterventionId], list[Path]], zip(*file_inputs)
    )
    page_counts = (_document_page_number(p) for p in files)

    def convert_all_with_retry(files: Iterable[Path]):
        return (
            (
                p,
                result
                if result is not None
                else _retry_scanning_failed_document(p, documentConvertor),
            )
            for p, result in cache.manualy_cache_batch_processing(
                cache_dd.get_yaml_file_for_pdf,
                cache_dd.cache_docling_doc_on_disk,
                cache_dd.load_docling_doc_from_cache,
                lambda it: map(
                    has_document_been_well_scanned,
                    documentConvertor.convert_all(
                        it, max_num_pages=150, raises_on_error=False
                    ),
                ),
                iter(files),
            )
        )

    def convert_with_debug(ids: list[InterventionId], files: list[Path]):
        result_iter = convert_all_with_retry(files)
        ids_iter = iter(ids)
        for max_duration in tqdm(
            page_counts,
            desc="vllm-scanned files",
            unit="file",
            total=len(file_inputs),
        ):
            print_log(
                f"Max duration for the current scan: {max_duration * timeout_per_page} s",
            )
            f, r = next(result_iter)
            id_ = next(ids_iter)
            yield (id_, f), r

    return list(convert_with_debug(ids, files))
