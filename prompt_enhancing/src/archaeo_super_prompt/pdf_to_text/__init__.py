from pathlib import Path
from sklearn.pipeline import FunctionTransformer
from ..types.pdfchunks import PDFChunkDataset, composePdfChunkDataset
from ..types.pdfpaths import (
    PDFPathDataset,
    buildPdfPathDataset,
    get_intervention_rows,
)

from .add_ocr import add_ocr_layer
from .smart_reading import extract_smart_chunks_from_pdf


def _ocr_transform(X: PDFPathDataset) -> PDFPathDataset:
    """
    Arguments:
        * X : A dataframe with "id" a column of Id and "filepath" a column of
    pathlib.Path
    """
    ids, paths = X["id"], X["filepath"]
    output_paths = add_ocr_layer([Path(p) for p in paths])
    return buildPdfPathDataset(t for t in zip(ids, output_paths, strict=True))


def OCR_Transformer():
    return FunctionTransformer(_ocr_transform)


def _text_extract(X: PDFPathDataset) -> PDFChunkDataset:
    return PDFChunkDataset(
        composePdfChunkDataset(
            extract_smart_chunks_from_pdf(pdf_path, intervention_id)
            for intervention_id, pdf_path in get_intervention_rows(X)
        )
    )


def TextExtractor():
    return FunctionTransformer(_text_extract)
