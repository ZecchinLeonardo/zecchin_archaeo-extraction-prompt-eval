"""Abstract data type for handling a dataset of read pdfs"""

from pandas import concat
from pandera.pandas import DataFrameModel
from pandera.typing import DataFrame, Series
from typing import Generator, Iterable, List, NewType, Tuple, TypedDict, Union, cast

from archaeo_super_prompt.signatures.input import (
    Chunk,
    ChunkHumanDescription,
    Filename,
    PDFSources,
)

from .intervention_id import InterventionId


# TODO: type check


class PDFChunkSetPerInterventionSchema(DataFrameModel):
    filename: Series[Filename]
    chunk_type: Series[str]
    chunk_page_position: Series[str]  # fraction: page number over total page number
    chunk_index: Series[int]
    chunk_content: Series[Chunk]


class PDFChunkDatasetSchema(PDFChunkSetPerInterventionSchema):
    id: Series[InterventionId]


PDFChunkPerInterventionDataset = NewType(
    "PDFChunkPerInterventionDataset", DataFrame[PDFChunkSetPerInterventionSchema]
)
PDFChunkDataset = NewType("PDFChunkDataset", DataFrame[PDFChunkDatasetSchema])


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


def composePdfChunkDataset(
    datasets: Union[Generator[PDFChunkDataset], Iterable[PDFChunkDataset]],
) -> PDFChunkDataset:
    return PDFChunkDataset(concat(datasets))


def buildPdfChunkDataset(chunks: List[PDFChunk]) -> PDFChunkDataset:
    return PDFChunkDataset(DataFrame(chunks))


def getExtractedPdfContent(
    dataset: PDFChunkPerInterventionDataset,
) -> PDFSources:
    """Let dataset be a set of chunks from several pdf files related to a
    single intervention. Computes the batch of chunk sources from this dataset

    The dataset can be partial if a selection of chunks in each files has
    already been carried out.
    """

    def items_for_pdf_source(fileChunks: PDFChunkDataset):
        def process_row(row_):
            TAG_TO_STRING = {
                "para": "Paragraph",
                "list_item": "List item",
                "table": "Table",
                "header": "Header",
            }
            DEFAULT_ITEM = "Unknown pdf item"
            row = PDFChunk(cast(PDFChunk, row_.to_dict()))
            tag_description = TAG_TO_STRING.get(row["chunk_type"], DEFAULT_ITEM)
            description = ChunkHumanDescription(
                f"Chunk {row['chunk_index']} ({tag_description} page {row['chunk_page_position']})"
            )
            return description, row["chunk_content"]

        return dict(process_row(row) for _, row in fileChunks.iterrows())

    return {
        Filename(cast(str, filename)): items_for_pdf_source(
            PDFChunkDataset(cast(PDFChunkDataset, fileChunks))
        )
        for filename, fileChunks in dataset.groupby("filename")
    }
