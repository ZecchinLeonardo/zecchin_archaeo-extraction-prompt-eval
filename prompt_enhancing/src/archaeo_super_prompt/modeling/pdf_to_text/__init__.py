"""PDF Ingestion layer with vision llm and chunking model."""

from pathlib import Path
from sklearn.pipeline import FunctionTransformer

from ...types.pdfchunks import PDFChunkDataset
from ...types.pdfpaths import (
    PDFPathDataset,
)

from .chunking import get_chunker, get_chunks, chunk_to_ds
from tqdm import tqdm
from . import stream_ocr_manual as vllm_scan_mod


def VLLM_Preprocessing(
    model: str,
    prompt: str,
    embedding_model_hf_id: str,
    incipit_only: bool,
    max_chunk_size: int = 512,
    allowed_timeout: int = 60 * 5,
):
    """First PDF ingestion layer for the pipeline. Include vision-llm scan and text chunking.

    This pipeline FunctionTransformer directly takes in input a batch of paths
    of PDF files to be ingested. It read the text with a vision-llm and output
    text chunks with being aware to the layout and a tokenization method to be
    provided.

    Arguments:
        model: the reference of the vision-llm to be called on the Ollama server
        prompt: a string to contextualize the ocr operation of the vision llm
        embedding_model_hf_id: the identifier on HuggingFace API of the embedding model, so its tokenizer can be fetched
        incipit_only: if only the first pages are scanned or all the document
        max_chunk_size: the maximum size of all text chunks
        allowed_timeout: the maximum duration for scanning text from one PDF page
    """
    allowed_timeout = allowed_timeout
    converter = vllm_scan_mod.converter(
        vllm_scan_mod.ollama_vlm_options(
            model, prompt, allowed_timeout=allowed_timeout
        )
    )
    chunker = get_chunker(embedding_model_hf_id, max_chunk_size)

    def transform(X: PDFPathDataset) -> PDFChunkDataset:
        conversion_results = vllm_scan_mod.process_documents(
            [(line["id"], Path(line["filepath"])) for _, line in X.iterrows()],
            converter,
            allowed_timeout,
            incipit_only
        )
        chunked_results = tqdm(((f, get_chunks(chunker, r)) for f, r in conversion_results), desc="Chunking read text", unit="chunked files", total=len(X))
        return chunk_to_ds(chunked_results, chunker)

    return FunctionTransformer(transform)
