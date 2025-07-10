from typing import Set, List, Tuple
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
import pandas as pd
from transformers import AutoTokenizer

from docling.chunking import HybridChunker
import functools as fnt

from pathlib import Path
from .types import CorrectlyConvertedDocument
from ..types.pdfchunks import PDFChunkDataset, PdfChunkDatasetSchema

EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"


def get_chunker(embed_model_id: str):
    # the tokenizer must be the same as the embedding model
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def get_chunks(chunker: HybridChunker, document: CorrectlyConvertedDocument) -> List[BaseChunk]:
    chunks = list(chunker.chunk(document.document))
    return chunks


def page_numbers_of_chunk(chunk: BaseChunk) -> Set[int]:
    return set(
        fnt.reduce(
            lambda acc_lst, item: acc_lst + [p.page_no for p in item.prov],
            chunk.meta.doc_items,
            [],
        )
    )


def chunk_types_of_chunk(chunk: BaseChunk) -> Set[str]:
    return set([item.label for item in chunk.meta.doc_items])


def chunk_to_ds(pairs: List[Tuple[Path,List[BaseChunk]]], chunker: HybridChunker) -> PDFChunkDataset:
    return PdfChunkDatasetSchema.validate(pd.concat(pd.DataFrame([{
        "filename": file.name,
        "chunk_type": chunk_types_of_chunk(chunk),
        "chunk_page_position": page_numbers_of_chunk(chunk),
        "chunk_index": chunk_idx,
        "chunk_content": chunker.contextualize(chunk),
    } for chunk_idx, chunk in enumerate(chunks_per_file)]) for file, chunks_per_file in pairs))
