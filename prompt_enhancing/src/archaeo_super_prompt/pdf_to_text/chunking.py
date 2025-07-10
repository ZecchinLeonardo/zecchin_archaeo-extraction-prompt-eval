from typing import Set
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from docling.chunking import HybridChunker
import functools as fnt

from .types import CorrectlyConvertedDocument
from ..types.pdfchunks import PDFChunkDataset, composePdfChunkDataset

EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"


def get_chunker(embed_model_id: str):
    # the tokenizer must be the same as the embedding model
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def get_chunks(chunker: HybridChunker, document: CorrectlyConvertedDocument):
    chunks = list(chunker.chunk(document.document))
    return chunks


def page_numbers_of_chunk(chunk: BaseChunk) -> Set[int]:
    d = chunk.export_json_dict()
    pages = set(
        fnt.reduce(
            lambda acc_lst, item: acc_lst + [p["page_no"] for p in item["prov"]],
            d["meta"]["doc_items"],
            [],
        )
    )
    return pages
