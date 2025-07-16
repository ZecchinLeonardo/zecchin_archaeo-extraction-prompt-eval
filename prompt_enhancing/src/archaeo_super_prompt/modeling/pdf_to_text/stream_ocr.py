"""Better OCR model with VLLM."""

from io import BytesIO
from pathlib import Path
from typing import (
    cast,
    Literal,
)
from collections.abc import Iterator, Callable, Iterable
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
from ...utils.cache import get_memory_for
from ...utils import cache

from ...config.debug_log import print_log


_PARALLEL_PAGE_NB = 2


def _document_page_number(file: Path) -> int:
    source_doc = pymupdf.open(file)
    page_count = source_doc.page_count
    source_doc.close()
    return page_count


def stream_document_pages(file: Path) -> tuple[Iterable[DocumentStream], int]:
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
        return DocumentStream(
            # intervention_id + fileanme + page_number + .pdf
            name=f"{file.parent.name}__{file.stem}__p{page_number}.pdf",
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
) -> list[CorrectlyConvertedDocument]:
    print_log("Retry scanning the document page per page...")
    pages_iter, page_number = stream_document_pages(doc)
    return [
        r
        for r in tqdm(
            map(
                has_document_been_well_scanned,
                docConverter.convert_all(
                    pages_iter, max_num_pages=1, raises_on_error=False
                ),
            ),
            desc="Individual page scan",
            unit="page",
            total=page_number,
        )
        if r is not None
    ]


def _vllm_cache_output(
    filepath: str,
    output: list[CorrectlyConvertedDocument] | None = None,
) -> list[CorrectlyConvertedDocument] | None:
    return cache.identity_function(filepath, output)


def _cached_convert_all(
    convert_all_func: Callable[
        [Iterable[Path]], Iterator[list[CorrectlyConvertedDocument]]
    ],
    files: list[tuple[InterventionId, Path]],
):
    """Run the conversion for all the documents, using cache if possible.

    If the vllm when the results are not cached, it save them in the cache
    after the computation.

    Arguments:
        convert_all_func: a function which should always return a valid \
ConversionResult (so failed results must be managed in this function
        files: a list of PDF files to be processed, related to their \
intervention id

    Return:
        An Iterable with (documentFilePath, CorrectlyConvertedDocument)
    """

    def normalized_path_str(p: Path):
        return str(p.resolve())

    return (
        ((id_, f), r)
        for id_, (f, r) in zip(
            (id_ for id_, _ in files),
            cache.escape_expensive_run_when_cached(
                _vllm_cache_output,
                get_memory_for("interim"),
                normalized_path_str,
                convert_all_func,
                (f for _, f in files),
            ),
        )
    )


def process_documents(
    file_inputs: list[tuple[InterventionId, Path]],
    documentConvertor: DocumentConverter,
    timeout_per_page: int,
) -> list[
    tuple[tuple[InterventionId, Path], list[CorrectlyConvertedDocument]]
]:
    """Convert the documents into text with Docling, using the given converter.

    Return:
    For each file, either a list of one docling document, if all the document
    can have been procesed at once, or a list of docling documents for each
    document page.
    """
    _, files = cast(tuple[list[InterventionId], list[Path]], zip(*file_inputs))
    page_counts = (_document_page_number(p) for p in files)

    def convert_all_with_retry(files: Iterable[Path]):
        files = list(files)
        for result, f in zip(
            map(
                has_document_been_well_scanned,
                documentConvertor.convert_all(
                    files, max_num_pages=150, raises_on_error=False
                ),
            ),
            files,
        ):
            if result is not None:
                yield [result]
            else:
                yield _retry_scanning_failed_document(f, documentConvertor)

    def convert_with_debug():
        print_log(
            f"Max duration for the current scan: {next(page_counts) * timeout_per_page} s",
        )
        for f, result in tqdm(
            _cached_convert_all(convert_all_with_retry, file_inputs),
            desc="vllm-scanned files",
            unit="file",
            total=len(file_inputs)
        ):
            yield f, result
            try:
                print_log(
                    f"Max duration for the current scan: {next(page_counts) * timeout_per_page} s",
                )
            except StopIteration:
                # edge-case for the last loop
                pass

    return list(convert_with_debug())
