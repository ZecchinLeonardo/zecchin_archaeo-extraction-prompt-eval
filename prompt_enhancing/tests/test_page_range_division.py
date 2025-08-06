"""Test the division of pages for a vlm processing by docling."""

from archaeo_super_prompt.modeling.pdf_to_text.stream_ocr_manual import (
    get_page_ranges, INCIPIT_MAX_PAGES
)

def test_page_range():
    """."""
    assert list(get_page_ranges(5, 2)) == [(1, 2), (3, 4), (5, 5)]
    assert list(get_page_ranges(5, 3)) == [(1, 3), (4, 5)]
    assert list(get_page_ranges(5, 1)) == [(i, i) for i in range(1, 6)]
    assert list(get_page_ranges(INCIPIT_MAX_PAGES, INCIPIT_MAX_PAGES, INCIPIT_MAX_PAGES)) == [(1, INCIPIT_MAX_PAGES)]
    assert list(get_page_ranges(2*INCIPIT_MAX_PAGES, INCIPIT_MAX_PAGES, INCIPIT_MAX_PAGES)) == [(1, INCIPIT_MAX_PAGES), (INCIPIT_MAX_PAGES+1, 2*INCIPIT_MAX_PAGES)]
    
