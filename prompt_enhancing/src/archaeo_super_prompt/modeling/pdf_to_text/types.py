from typing import NewType, Optional
from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.document import DoclingDocument


CorrectlyConvertedDocument = NewType(
    "CorrectlyConvertedDocument", DoclingDocument
)


def has_document_been_well_scanned(
    doc: ConversionResult,
) -> Optional[CorrectlyConvertedDocument]:
    """Return the docling document typed as validated if the status of the
    conversion is at least partially successfull, else return None
    """
    if (
        doc.status == ConversionStatus.SUCCESS
        or doc.status == ConversionStatus.PARTIAL_SUCCESS
    ):
        return CorrectlyConvertedDocument(doc.document)
    return None
