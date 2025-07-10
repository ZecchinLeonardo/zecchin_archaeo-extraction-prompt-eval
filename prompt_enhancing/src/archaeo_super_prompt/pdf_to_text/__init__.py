from pathlib import Path
from sklearn.pipeline import FunctionTransformer

from ..types.pdfchunks import PDFChunkDataset
from ..types.pdfpaths import (
    PDFPathDataset,
)

from .chunking import get_chunker, get_chunks, chunk_to_ds
from . import stream_ocr as vllm_scan_mod


def VLLM_Preprocessing(
    model: str,
    prompt: str,
    embedding_model_hf_id: str,
    allowed_timeout: int = 60 * 5,
):
    """Arguments:
    * embedding_model_hf_id: the identifier on HuggingFace API of the embedding model, so its tokenizer can be fetched
    """
    allowed_timeout = allowed_timeout
    converter = vllm_scan_mod.converter(
        vllm_scan_mod.ollama_vlm_options(model, prompt, allowed_timeout=allowed_timeout)
    )
    chunker = get_chunker(embedding_model_hf_id)

    def transform(X: PDFPathDataset) -> PDFChunkDataset:
        conversion_results = vllm_scan_mod.process_documents(
            [(line["id"], Path(line["filepath"])) for _, line in X.iterrows()],
            converter,
            allowed_timeout,
        )
        chunked_results = [(f, get_chunks(chunker, r)) for f, r in conversion_results]

        return chunk_to_ds(chunked_results, chunker)

    return FunctionTransformer(transform)
