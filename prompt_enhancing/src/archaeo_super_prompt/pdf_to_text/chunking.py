from typing import Set, List, Tuple, cast
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DocItem, ProvenanceItem
import pandas as pd
from transformers import AutoTokenizer

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
import functools as fnt

from pathlib import Path
from .types import CorrectlyConvertedDocument
from ..types.intervention_id import InterventionId
from ..types.pdfchunks import PDFChunkDataset, PDFChunkDatasetSchema

EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"


def get_chunker(embed_model_id: str):
    # the tokenizer must be the same as the embedding model
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def _get_doc_items(chunk: BaseChunk) -> List[DocItem]:
    return cast(List[DocItem], chunk.meta.doc_items)  # type: ignore


def _set_doc_items(chunk: BaseChunk, doc_items: List[DocItem]):
    # not pure
    chunk.meta.doc_items = doc_items  # type: ignore


def get_chunks(
    chunker: HybridChunker, documents: List[CorrectlyConvertedDocument]
) -> List[BaseChunk]:
    if not documents:
        return []
    if len(documents) == 1:
        return list(chunker.chunk(dl_doc=documents[0]))

    def adapt_page_numbers(chunk: BaseChunk, page_number: int):
        new_chunk = chunk.model_copy(deep=True)

        def adapt_page_number_for_doc_item(item: DocItem):
            def adapt_page_number_for_prov(prov: ProvenanceItem):
                new_prov = prov.model_copy(deep=True)
                new_prov.page_no = page_number
                return new_prov

            new_item = item.model_copy(deep=True)
            new_item.prov = list(map(adapt_page_number_for_prov, new_item.prov))
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
        for page_nb, chunks in enumerate(map(lambda d: chunker.chunk(dl_doc=d), documents))
    )
    chunks = fnt.reduce(
        lambda chunk_lst, per_page_chunk_pack: chunk_lst + per_page_chunk_pack,
        per_page_chunk_packs,
        cast(List[BaseChunk], []),
    )
    return chunks


def _page_numbers_of_chunk(chunk: BaseChunk) -> Set[int]:
    return set(
        fnt.reduce(
            lambda acc_lst, item: list(acc_lst) + list(p.page_no for p in item.prov),
            _get_doc_items(chunk),
            cast(List[int], []),
        )
    )


def _chunk_types_of_chunk(chunk: BaseChunk) -> Set[str]:
    return set([item.label for item in _get_doc_items(chunk)])


def chunk_to_ds(
    pairs: List[Tuple[Tuple[InterventionId, Path], List[BaseChunk]]],
    chunker: HybridChunker,
) -> PDFChunkDataset:
    return PDFChunkDataset(
        PDFChunkDatasetSchema.validate(
            pd.concat(
                (
                    pd.DataFrame(
                        [
                            {
                                "id": int(id_),
                                "filename": file.name,
                                "chunk_type": list(_chunk_types_of_chunk(chunk)),
                                "chunk_page_position": list(
                                    _page_numbers_of_chunk(chunk)
                                ),
                                "chunk_index": chunk_idx,
                                "chunk_content": chunker.contextualize(chunk),
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
