from pathlib import Path
from typing import Tuple, cast

from archaeo_super_prompt.types.intervention_id import InterventionId

from ..types.pdfchunks import PDFChunkDataset, composePdfChunkDataset
from ..types.pdfpaths import (
    PDFPathDataset,
    buildPdfPathDataset,
    get_intervention_rows,
)

from ..dataset.load import MagohDataset
from .add_ocr import add_ocr_layer
from .smart_reading import extract_smart_chunks_from_pdf

class OCR_Transformer:
    def fit(self, X: PDFPathDataset, y=None):
        X = X
        y = y
        return self

    def transform(self, X: PDFPathDataset) -> PDFPathDataset:
        """
        Arguments:
            * X : A dataframe with "id" a column of Id and "filepath" a column of
        pathlib.Path
        """
        ids, paths = cast(
            Tuple[Tuple[InterventionId, ...], Tuple[Path, ...]],
            zip(*get_intervention_rows(X), strict=True),
        )
        output_paths = add_ocr_layer(list(paths))
        return buildPdfPathDataset(t for t in zip(ids, output_paths, strict=True))


class TextExtractor:
    def fit(self, X: PDFPathDataset, targets: MagohDataset):
        X = X
        targets = targets
        return self

    def transform(self, X: PDFPathDataset) -> PDFChunkDataset:
        return PDFChunkDataset(
            composePdfChunkDataset(
                extract_smart_chunks_from_pdf(pdf_path, intervention_id)
                for intervention_id, pdf_path in get_intervention_rows(X)
            )
        )
