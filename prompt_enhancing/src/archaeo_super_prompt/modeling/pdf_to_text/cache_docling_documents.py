"""Manage the manual cache of the docling documents extracted from the pdf.

The Docling documents wear all the information a VLLM can extract from a pdf
document. Then, we define in this module how to cache this output to avoid to
recompute it with a VLLM call.
"""

from pathlib import Path
from typing import NamedTuple

from docling_core.types.doc.document import DoclingDocument

from archaeo_super_prompt.types.intervention_id import InterventionId

from .types import CorrectlyConvertedDocument
from ...utils import cache

DOC_DOC_SUBDIR = "pdf_scans"




class ArtificialPDFData(NamedTuple):
    """Data for saving data about a bufferized PDF document."""

    intervention_id: InterventionId
    filestem: str
    page_number: int


def _get_yaml_file_for_artificial_pdf(pdf_data: ArtificialPDFData) -> Path:
    """Return a yaml file to cache the extracted document from a pdf buffer."""
    return (
        cache.get_cache_dir_for("interim", DOC_DOC_SUBDIR)
        / str(pdf_data.intervention_id)
    ) / f"{pdf_data.filestem}.{pdf_data.page_number}.docling.yaml"

def _get_yaml_file_for_saved_pdf(source_pdf_path: Path) -> Path:
    cache_docling_doc_path = (
        Path(source_pdf_path.parent.name)
        / f"{source_pdf_path.stem}.docling.yaml"
    )
    return (
        cache.get_cache_dir_for("interim", DOC_DOC_SUBDIR)
        / cache_docling_doc_path
    )

def get_yaml_file_for_pdf(source_pdf_path: Path | ArtificialPDFData) -> Path:
    """Return a yaml file in which the extracted docling document can be cached."""
    if isinstance(source_pdf_path, ArtificialPDFData):
        return _get_yaml_file_for_artificial_pdf(source_pdf_path)
    return _get_yaml_file_for_saved_pdf(source_pdf_path)

def cache_docling_doc_on_disk(
    docling_document: CorrectlyConvertedDocument | None, file_path: Path
):
    """Save the docling document in the given yaml file.

    If the scanning has failed (most of the time for timeout reason), then None
    is input and an empty file will be saved, so, even if the execution has
    failed, it will not be executed again as it is assumed it will fail again.
    """
    directory = file_path.parent
    if not directory.exists():
        directory.mkdir(parents=True)
    if docling_document is None:
        return file_path.touch()
    return docling_document.save_as_yaml(file_path)


def load_docling_doc_from_cache(
    file_path_in_cache: Path,
) -> CorrectlyConvertedDocument | None:
    """Reload a cached docling document from its cached yaml file."""

    def is_file_empty(fp: Path):
        return fp.stat().st_size == 0

    if is_file_empty(file_path_in_cache):
        return None  # empty file: the document creation failed
    return CorrectlyConvertedDocument(
        DoclingDocument.load_from_yaml(file_path_in_cache)
    )
