from pathlib import Path
from typing import cast
from sklearn.pipeline import FunctionTransformer
from tqdm import tqdm

from archaeo_super_prompt.debug_log import print_warning

from ..types.intervention_id import InterventionId
from ..types.pdfchunks import PDFChunkDataset, composePdfChunkDataset
from ..types.pdfpaths import (
    PDFPathDataset,
    buildPdfPathDataset,
)

from .add_ocr import add_ocr_layer
from .smart_reading import (
    UnreadableSourceSetError,
    extract_smart_chunks_from_pdfs_of_intervention,
)


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
    def readASet(pdf_path: PDFPathDataset, intervention_id: InterventionId):
        try:
            return extract_smart_chunks_from_pdfs_of_intervention(
                pdf_path["filepath"], intervention_id
            )
        except UnreadableSourceSetError:
            print_warning(
                f"Impossible to read text from the given source set for intervention n°{intervention_id}"
            )
            return None

    return PDFChunkDataset(
        composePdfChunkDataset(
            cast(PDFChunkDataset, elt)
            for elt in filter(
                lambda elt: elt is not None,
                (
                    readASet(
                        cast(PDFPathDataset, pdf_path),
                        InterventionId(cast(int, intervention_id)),
                    )
                    for intervention_id, pdf_path in tqdm(
                        X.groupby("id"),
                        desc="Layout-Read PDF sets",
                        leave=False,
                        unit="intervention",
                    )
                ),
            )
        )
    )


def TextExtractor():
    return FunctionTransformer(_text_extract)
