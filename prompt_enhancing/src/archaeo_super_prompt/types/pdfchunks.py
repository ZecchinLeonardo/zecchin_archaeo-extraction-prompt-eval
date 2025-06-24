"""Abstract data type for handling a dataset of read pdfs"""

from pandas import concat
from pandera.pandas import DataFrameModel
from pandera.typing import DataFrame, Series
from typing import Generator, Iterable, List, NewType, TypedDict, Union, cast

from archaeo_super_prompt.signatures.input import (
    Chunk,
    ChunkHumanDescription,
    Filename,
    PDFSources,
)

from .intervention_id import InterventionId

# TODO: stronger type checking


class PDFChunkSetPerInterventionSchema(DataFrameModel):
    filename: Series[Filename]
    chunk_type: Series[str]
    chunk_page_position: Series[str]  # fraction: page number over total page number
    chunk_index: Series[int]
    chunk_content: Series[Chunk]


class PDFChunkDatasetSchema(PDFChunkSetPerInterventionSchema):
    id: Series[InterventionId]


PDFChunkDataset = NewType("PDFChunkDataset", DataFrame[PDFChunkDatasetSchema])


class PDFChunkPerInterventionDataset:
    """
    DataFrame class wrapper to customize the auto-displaying from tracing tools
    such as mlflow
    """

    TAG_TO_STRING = {
        "para": "Paragraph",
        "list_item": "List item",
        "table": "Table",
        "header": "Header",
    }
    DEFAULT_ITEM = "Unknown pdf item"

    def __init__(self, data: DataFrame[PDFChunkSetPerInterventionSchema]) -> None:
        self.data = data

    def getExtractedPdfContent(self) -> PDFSources:
        """Let dataset be a set of chunks from several pdf files related to a
        single intervention. Computes the batch of chunk sources from this dataset

        The dataset can be partial if a selection of chunks in each files has
        already been carried out.
        """

        def items_for_pdf_source(fileChunks: PDFChunkDataset):
            def process_row(row_):
                row = PDFChunk(cast(PDFChunk, row_.to_dict()))
                tag_description = PDFChunkPerInterventionDataset.TAG_TO_STRING.get(
                    row["chunk_type"], PDFChunkPerInterventionDataset.DEFAULT_ITEM
                )
                description = ChunkHumanDescription(
                    f"Chunk {row['chunk_index']} ({tag_description} page {row['chunk_page_position']})"
                )
                return description, row["chunk_content"]

            return dict(process_row(row) for _, row in fileChunks.iterrows())

        return {
            Filename(cast(str, filename)): items_for_pdf_source(
                PDFChunkDataset(cast(PDFChunkDataset, fileChunks))
            )
            for filename, fileChunks in self.data.groupby("filename")
        }

    def __str__(self) -> str:
        msg = ""
        for _, chunk in self.data.iterrows():
            msg += f"`%% {chunk['filename']} | Page {chunk['chunk_page_position']} ({PDFChunkPerInterventionDataset.TAG_TO_STRING.get(chunk['chunk_type'], PDFChunkPerInterventionDataset.DEFAULT_ITEM)}) %%`\n\n"
            msg += chunk["chunk_content"] + "\n" * 2
            msg += "`" + "-" * 60 + "`\n\n"
        return msg


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
"""NB: this type of row is unnormalized for a memory-efficient processing but
this might not be an issue in our pipeline, as the datasets are not huge and
the time processing wille be negligible next to the LLM and Embedding model
inferences
"""


def composePdfChunkDataset(
    datasets: Union[Generator[PDFChunkDataset], Iterable[PDFChunkDataset]],
) -> PDFChunkDataset:
    return PDFChunkDataset(cast(DataFrame, concat(datasets)))


def buildPdfChunkDataset(chunks: List[PDFChunk]) -> PDFChunkDataset:
    return PDFChunkDataset(DataFrame(chunks))
