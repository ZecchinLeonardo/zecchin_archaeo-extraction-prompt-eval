"""Utility functions to divide the pages of a PDF document into slices."""

from docling.datamodel.settings import PageRange


def get_page_ranges(
    doc_page_number: int,
    page_batch_size: int,
    border_page_nb: int | None = None,
) -> list[PageRange]:
    """Divide a number of pages into batch intervals.

    If only the header and the footer of the document are wanted, then
    only divide the first pages and the last pages into batch intervals. Set
    the argument border_page_nb to trigger such a behaviour.

    The number of page in a batch is set according to the number of page
    the remote LLM is able to process in parallel.

    Arguments:
        doc_page_number: the total number of pages in the document
        page_batch_size: the number of pages in a slice
        border_page_nb: if given, only keep this number of page from the start and from the end (so 2*border_page_nb) will be processed with the output ranges
    """

    def split_into_batch_page_range(start_page: int, end_page: int):
        return [
            (i, min(i + page_batch_size - 1, end_page))
            for i in range(
                start_page,
                min(end_page, doc_page_number) + 1,
                page_batch_size,
            )
        ]

    def get_start_and_end_pages(border_page_nb: int):
        if doc_page_number < 2 * border_page_nb:
            return split_into_batch_page_range(1, doc_page_number)
        return [
            *split_into_batch_page_range(1, border_page_nb),
            *split_into_batch_page_range(
                doc_page_number - border_page_nb + 1, doc_page_number
            ),
        ]

    if border_page_nb is not None:
        return get_start_and_end_pages(border_page_nb)
    return split_into_batch_page_range(1, doc_page_number)
