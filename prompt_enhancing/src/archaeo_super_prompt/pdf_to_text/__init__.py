from pathlib import Path
from typing import cast
from sklearn.pipeline import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


from ..debug_log import print_warning

from ..types.intervention_id import InterventionId
from ..types.pdfchunks import PDFChunkDataset, PdfChunkDatasetSchema, composePdfChunkDataset
from ..types.pdfpaths import (
    PDFPathDataset,
    buildPdfPathDataset,
)

from .add_ocr import add_ocr_layer
from .chunking import get_chunker, get_chunks, chunk_to_ds
from .smart_reading import (
    UnreadableSourceSetError,
    extract_smart_chunks_from_pdfs_of_intervention,
)
from . import stream_ocr as vllm_scan_mod


class VLLM_Preprocessing(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        model: str,
        second_model: str,
        prompt: str,
        embedding_model_hf_id: str,
        allowed_timeout: int = 60 * 5,
    ):
        """Arguments:
        * embedding_model_hf_id: the identifier on HuggingFace API of the embedding model, so its tokenizer can be fetched
        """
        self._allowed_timeout = allowed_timeout
        self._converter = vllm_scan_mod.converter(
            vllm_scan_mod.ollama_vlm_options(
                model, prompt, allowed_timeout=allowed_timeout
            )
        )
        self._second_converter = vllm_scan_mod.converter(
            vllm_scan_mod.ollama_vlm_options(
                second_model, prompt, allowed_timeout=allowed_timeout
            )
        )
        self._chunker = get_chunker(embedding_model_hf_id)

    def transform(self, X: PDFPathDataset) -> PDFChunkDataset:
        conversion_results = vllm_scan_mod.process_documents(
            [Path(p) for p in X["filepath"].to_list()],
            self._converter,
            self._second_converter,
            self._allowed_timeout,
        )
        chunked_results = [(f, get_chunks(self._chunker, r))
                           for f, r in conversion_results]

        return chunk_to_ds(chunked_results, self._chunker)


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
                f"Impossible to read text from the given source set for intervention n°{
                    intervention_id
                }"
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
