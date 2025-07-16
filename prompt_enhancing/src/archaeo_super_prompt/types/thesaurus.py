"""Global types related to thesaurus handling."""

from collections.abc import Callable

Thesaurus = tuple[int, str]
ThesaurusProvider = Callable[[], list[Thesaurus]]
