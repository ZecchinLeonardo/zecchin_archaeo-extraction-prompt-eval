from typing import NewType, TypeGuard
from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import ConversionStatus


CorrectlyConvertedDocument = NewType("CorrectlyConvertedDocument", ConversionResult)

def has_document_been_well_scanned(
    doc: ConversionResult,
) -> TypeGuard[CorrectlyConvertedDocument]:
    return (
        doc.status == ConversionStatus.SUCCESS
        or doc.status == ConversionStatus.PARTIAL_SUCCESS
    )
