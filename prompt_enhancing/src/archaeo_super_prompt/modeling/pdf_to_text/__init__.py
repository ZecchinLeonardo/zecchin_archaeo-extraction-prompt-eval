"""PDF Ingestion layer with vision llm and chunking model."""

from pathlib import Path
from typing import Literal, override

from tqdm import tqdm

from ...types.pdfchunks import PDFChunkDataset
from ...types.pdfpaths import (
    PDFPathDataset,
)
from ..types.base_transformer import BaseTransformer
from . import chunking as vllm_doc_chunk_mod
from . import stream_ocr_manual as vllm_scan_mod


class VLLM_Preprocessing(BaseTransformer):
    """First PDF ingestion layer for the pipeline. Include vision-llm scan and text chunking.

    This pipeline FunctionTransformer directly takes in input a batch of paths
    of PDF files to be ingested. It read the text with a vision-llm and output
    text chunks with being aware to the layout and a tokenization method to be
    provided.
    """

    def __init__(
        self,
        vlm_provider: Literal["ollama", "vllm", "openai"],
        vlm_model_id: str,
        prompt: str,
        embedding_model_hf_id: str,
        incipit_only: bool,
        max_chunk_size: int = 512,
        allowed_timeout: int = 60 * 5,
    ):
        """Provide the vlm model credentials and other parametres.

        Arguments:
            vlm_provider: the remote service to connect to
            vlm_model_id: the reference of the vision-llm to be called on the Ollama server
            prompt: a string to contextualize the ocr operation of the vision llm
            embedding_model_hf_id: the identifier on HuggingFace API of the embedding model, so its tokenizer can be fetched
            incipit_only: if only the first pages are scanned or all the document
            max_chunk_size: the maximum size of all text chunks
            allowed_timeout: the maximum duration for scanning text from one PDF page

        Environment variable:
            The VLM_HOST_URL env var must be set like this :
            http://localhost:8005 
        """
        # store the parameters for logging
        self.vlm_provider = vlm_provider
        self.vlm_model_id = vlm_model_id
        self.prompt = prompt
        self.embedding_model_hf_id = embedding_model_hf_id
        self.incipit_only = incipit_only
        self.max_chunk_size = max_chunk_size
        self.allowed_timeout = allowed_timeout

        self._chunker = vllm_doc_chunk_mod.get_chunker(
            embedding_model_hf_id, max_chunk_size
        )

    @override
    def transform(self, X: PDFPathDataset) -> PDFChunkDataset:
        # instantiate the converter at runtime so the environment variable of
        # the endpoint of the vlm is not cached if the instance of the
        # Transformer is cached by joblib, as in standard sklearn workflows
        converter = vllm_scan_mod.converter(
                    vllm_scan_mod.vllm_vlm_options(
                        self.vlm_model_id, self.prompt, allowed_timeout=self.allowed_timeout
                    )
                    if self.vlm_provider != "ollama"
                    else vllm_scan_mod.ollama_vlm_options(
                        self.vlm_model_id, self.prompt, allowed_timeout=self.allowed_timeout
                    )
                )
        conversion_results = vllm_scan_mod.process_documents(
            [(line["id"], Path(line["filepath"])) for _, line in X.iterrows()],
            converter,
            self.incipit_only,
        )
        chunked_results = iter(
            tqdm(
                (
                    (f, vllm_doc_chunk_mod.get_chunks(self._chunker, r))
                    for f, r in conversion_results
                ),
                desc="Chunking read text",
                unit="chunked files",
                total=len(X),
            )
        )
        return vllm_doc_chunk_mod.chunk_to_ds(chunked_results, self._chunker)
