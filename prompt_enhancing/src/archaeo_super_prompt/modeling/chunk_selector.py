"""Utils to select boundaries page of interest in documents."""

import math
import functools as fnt
from typing import cast

from ..types.pdfchunks import PDFChunkPerInterventionDataset


def _get_reasonable_page_number(chunkDataset: PDFChunkPerInterventionDataset):
    MAX_SELECTABLE_PAGE_NUMBER = 3
    total_page_number = max(
        fnt.reduce(
            lambda flat, lst: flat + lst,
            cast(
                list[list[int]],
                chunkDataset.data["chunk_page_position"].to_list(),
            ),
            cast(list[int], []),
        )
    )
    max_selected_page_number = min(
        MAX_SELECTABLE_PAGE_NUMBER, math.ceil(0.1 * total_page_number)
    )
    return max_selected_page_number, total_page_number


def select_incipit(chunkDataset: PDFChunkPerInterventionDataset):
    """Select only the chunks of the first pages of the document."""
    max_selected_page_number, _ = _get_reasonable_page_number(chunkDataset)
    return PDFChunkPerInterventionDataset(
        chunkDataset.data[
            chunkDataset.data["chunk_page_position"].apply(min)
            < max_selected_page_number
        ]
    )


def select_end_pages(chunkDataset: PDFChunkPerInterventionDataset):
    """Select only the chunks of the last pages of the document."""
    max_selected_page_number, total_page_number = _get_reasonable_page_number(
        chunkDataset
    )
    return PDFChunkPerInterventionDataset(
        chunkDataset.data[
            chunkDataset.data["chunk_page_position"].apply(max)
            > total_page_number - max_selected_page_number
        ]
    )
