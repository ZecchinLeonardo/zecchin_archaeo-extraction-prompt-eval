"""Scanned document splitting into text chunks with layout metadata."""

import functools as fnt
from collections.abc import Iterable, Iterator
from functools import reduce
from pathlib import Path
from typing import cast

import pandas as pd
from docling.datamodel.settings import PageRange
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc.document import DocItem
from transformers import AutoTokenizer

from ...types.intervention_id import InterventionId
from ...types.pdfchunks import PDFChunkDataset, PDFChunkDatasetSchema
from ...utils.cache import get_memory_for
from .types import CorrectlyConvertedDocument

EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"


def get_chunker(embed_model_id: str, max_chunk_size: int):
    """Return a Docling Chunker model according to the tokenizer of one embedding model.

    This tokenizer is fast even on the CPU, but must be fetch from the
    HuggingFace's repositories.
    """
    # the tokenizer must be the same as the embedding model
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
        max_tokens=max_chunk_size,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


@get_memory_for("interim").cache
def get_chunks(
    chunker: HybridChunker,
    document: Iterator[tuple[PageRange, CorrectlyConvertedDocument]],
) -> list[BaseChunk]:
    """Extracts a list of labeled chunks through all the pages of the document.

    Arguments:
        chunker: the chunker model to chunk according to the layout and the \
tokenization
        document: the document or a list of documents for each page
    """
    return reduce(
        lambda flatten, d: ([*flatten, *chunker.chunk(dl_doc=d)]),
        (d for _, d in document),
        cast(list[BaseChunk], []),
    )


def _get_doc_items(chunk: BaseChunk) -> list[DocItem]:
    return cast(list[DocItem], chunk.meta.doc_items)  # type: ignore


def _page_numbers_of_chunk(chunk: BaseChunk) -> set[int]:
    return set(
        fnt.reduce(
            lambda acc_lst, item: list(acc_lst)
            + list(p.page_no for p in item.prov),
            _get_doc_items(chunk),
            cast(list[int], []),
        )
    )


def _chunk_types_of_chunk(chunk: BaseChunk) -> set[str]:
    return set([str(item.label) for item in _get_doc_items(chunk)])


def chunk_to_ds(
    pairs: Iterable[tuple[tuple[InterventionId, Path], list[BaseChunk]]],
    chunker: HybridChunker,
) -> PDFChunkDataset:
    """Gather the list of labeled chunks into a dataframe for all the document batch."""
    return PDFChunkDataset(
        PDFChunkDatasetSchema.validate(
            pd.concat(
                (
                    pd.DataFrame(
                        [
                            {
                                "id": int(id_),
                                "filename": file.name,
                                "chunk_type": list(
                                    _chunk_types_of_chunk(chunk)
                                ),
                                "chunk_page_position": list(
                                    _page_numbers_of_chunk(chunk)
                                ),
                                "chunk_index": chunk_idx,
                                "chunk_embedding_content": chunker.contextualize(
                                    chunk
                                ),
                                "chunk_content": chunk.text,
                            }
                            for chunk_idx, chunk in enumerate(chunks_per_file)
                        ]
                    )
                    for (id_, file), chunks_per_file in pairs
                ),
                ignore_index=True,
            )
        )
    )
