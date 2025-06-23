"""Abstract data type for handling a dataset of read pdfs"""

from functools import reduce
from pandas import DataFrame, Series, concat
from typing import Generator, Iterable, List, NewType, TypedDict, Union, cast

from archaeo_super_prompt.signatures.input import (
    Chunk,
    ChunkHumanDescription,
    Filename,
    PDFSources,
)

from .intervention_id import InterventionId


# TODO: add filename
PDFChunkDataset = NewType("PDFChunkDataset", DataFrame)

"""NB: this type of row is unnormalized for a memory-efficient processing but
this might not be an issue in our pipeline, as the datasets are not huge and
the time processing wille be negligible next to the LLM and Embedding model
inferences
"""
PDFChunk = TypedDict(
    "PDFChunk",
    {
        "id": InterventionId,
        "filename": Filename,
        "chunk_type": str,
        "chunk_page_position": str,  # fraction: page number over total page number
        "chunk_index": int,
        "chunk_content": Chunk,
    },
)


def _get_intervention_ids(ds: PDFChunkDataset) -> Series[InterventionId]:
    return ds["id"]


def composePdfChunkDataset(
    datasets: Union[Generator[PDFChunkDataset], Iterable[PDFChunkDataset]],
) -> PDFChunkDataset:
    return PDFChunkDataset(concat(datasets))


def buildPdfChunkDataset(chunks: List[PDFChunk]) -> PDFChunkDataset:
    return PDFChunkDataset(DataFrame(chunks))


def getExtractedPdfContent(
    dataset: PDFChunkDataset, intervention_id: InterventionId
) -> PDFSources:
    def add_to_dict(acc_d: PDFSources, row_: Series):
        row = PDFChunk(cast(PDFChunk, row_.to_dict()))
        filename = Filename(row["filename"])
        if filename not in acc_d:
            acc_d[filename] = {}
        description = ChunkHumanDescription(
            f"Chunk {row['chunk_index']} ({row['chunk_type']} page {row['chunk_page_position']})"
        )
        acc_d[filename][description] = row["chunk_content"]
        return acc_d

    return reduce(
        add_to_dict,
        (
            row
            for _, row in dataset[
                _get_intervention_ids(dataset) == intervention_id
            ].iterrows()
        ),
        dict(),
    )
