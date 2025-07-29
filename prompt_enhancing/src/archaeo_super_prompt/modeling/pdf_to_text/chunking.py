"""Scanned document splitting into text chunks with layout metadata."""

from collections.abc import Sequence, Iterable
from typing import cast
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc.document import DocItem, ProvenanceItem
import pandas as pd
from transformers import AutoTokenizer

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
import functools as fnt

from pathlib import Path
from .types import CorrectlyConvertedDocument
from ...types.intervention_id import InterventionId
from ...types.pdfchunks import PDFChunkDataset, PDFChunkDatasetSchema
from ...utils.cache import get_memory_for

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


def _get_doc_items(chunk: BaseChunk) -> list[DocItem]:
    return cast(list[DocItem], chunk.meta.doc_items)  # type: ignore


def _set_doc_items(chunk: BaseChunk, doc_items: list[DocItem]):
    # not pure
    chunk.meta.doc_items = doc_items  # type: ignore

@get_memory_for("interim").cache
def get_chunks(
    chunker: HybridChunker,
    document: CorrectlyConvertedDocument
    | Sequence[CorrectlyConvertedDocument | None],
) -> list[BaseChunk]:
    """Extracts a list of labeled chunks through all the pages of the document.

    Arguments:
        chunker: the chunker model to chunk according to the layout and the \
tokenization
        document: the document or a list of documents for each page
    """ 
    if not isinstance(document, Sequence):
        return list(chunker.chunk(dl_doc=document))
    if not document:
        return []

    def adapt_page_numbers(chunk: BaseChunk, page_number: int):
        new_chunk = chunk.model_copy(deep=True)

        def adapt_page_number_for_doc_item(item: DocItem):
            def adapt_page_number_for_prov(prov: ProvenanceItem):
                new_prov = prov.model_copy(deep=True)
                new_prov.page_no = page_number
                return new_prov

            new_item = item.model_copy(deep=True)
            new_item.prov = list(
                map(adapt_page_number_for_prov, new_item.prov)
            )
            return new_item

        _set_doc_items(
            new_chunk,
            list(
                map(
                    adapt_page_number_for_doc_item,
                    _get_doc_items(new_chunk),
                )
            ),
        )
        return new_chunk

    per_page_chunk_packs = (
        [adapt_page_numbers(chunk, page_nb) for chunk in chunks]
        for page_nb, chunks in (
            (pnb, chunker.chunk(dl_doc=d))
            for pnb, d in enumerate(document)
            if d is not None
        )
    )
    chunks = fnt.reduce(
        lambda chunk_lst, per_page_chunk_pack: chunk_lst + per_page_chunk_pack,
        per_page_chunk_packs,
        cast(list[BaseChunk], []),
    )
    return chunks


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
