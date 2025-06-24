import math

from ..types.pdfchunks import PDFChunkPerInterventionDataset


def select_incipit(chunkDataset: PDFChunkPerInterventionDataset):
    MAX_SELECTABLE_PAGE_NUMBER = 3
    max_page_number = int(
        chunkDataset.data["chunk_page_position"][0].rstrip().split("/")[1]
    )
    max_selected_page = min(
        MAX_SELECTABLE_PAGE_NUMBER, math.ceil(0.1 * max_page_number)
    )
    return PDFChunkPerInterventionDataset(
        chunkDataset.data[
            chunkDataset.data["chunk_page_position"].apply(
                lambda s: int(s.rstrip().split("/")[0])
            )
            < max_selected_page
        ]
    )

def select_end_pages(chunkDataset: PDFChunkPerInterventionDataset):
    # TODO: refactor
    MAX_SELECTABLE_PAGE_NUMBER = 3
    max_page_number = int(
        chunkDataset.data["chunk_page_position"][0].rstrip().split("/")[1]
    )
    max_selected_page = min(
        MAX_SELECTABLE_PAGE_NUMBER, math.ceil(0.1 * max_page_number)
    )
    return PDFChunkPerInterventionDataset(
        chunkDataset.data[
            chunkDataset.data["chunk_page_position"].apply(
                lambda s: int(s.rstrip().split("/")[0])
            )
            > max_page_number - max_selected_page
        ]
    )
