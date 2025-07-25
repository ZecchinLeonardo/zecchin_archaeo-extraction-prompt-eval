"""Identification of thesaurus with fuzzymatching in text chunks."""

from collections.abc import Iterable

from fuzzysearch import find_near_matches, Match
from thefuzz import fuzz

from .types import CompleteEntity, ThesaurusProvider


def extended_expression(content: str, match: Match) -> str:
    """Return the extended expression around a given match.

    Examples:
    "WE ARE IN PONTEDERA", "PONTE" -> "PONTEDERA"
    "WE ARE IN AN APPARTEMENT", "PART" -> "APPARTEMENT"
    "WE ARE IN AN APPARTEMENT", "APPARTEMENT" -> "APPARTEMENT"
    "I am working for the Soprintendenza Archeologica della Toscana", "Soprintendenza Archeologica della Toscana" -> "Soprintendenza Archeologica della Toscana"
    "I am working for the Soprintendenza Archeologica della Toscana", "intendenza Archeologica della Toscana" -> "Soprintendenza Archeologica della Toscana"
    """
    content_length = len(content)

    extended_start = match.start
    if content[extended_start].isalnum():
        while extended_start > 0 and content[extended_start - 1].isalnum():
            extended_start -= 1

    extended_end = match.end
    if content[extended_end - 1].isalnum():
        while (
            extended_end < content_length and content[extended_end].isalnum()
        ):
            extended_end += 1
    print(content[extended_start:extended_end])
    return content[extended_start:extended_end]


def filter_occurences(
    content: str, thesaurus_value: str, matches: list[Match]
) -> list[Match]:
    """Keep the matches whose extended expression still match with the thesarusus value.

    For example, if "PART" is detected in the content "WE ARE IN AN APPARTEMENT", then this match will be excluded.
    """
    print(thesaurus_value, ":")

    def filter_empty_word_matches(matches: list[Match]):
        return [m for m in matches if m.matched != ""]

    f = [
        match
        for match in filter_empty_word_matches(matches)
        # the levenstein distance will augment if the extended_expression is
        # too much longer, so the ratio will decrease
        if fuzz.ratio(extended_expression(content, match), thesaurus_value)
        > 80
    ]
    return f


def extract_from_content(
    content: str,
    entity_set: list[CompleteEntity],
    wanted_entities: list[tuple[int, str]],
):
    """We expect the wanted entities and the content to be normalized."""
    if not entity_set:
        return None
    return set(
        thesaurus_id
        for thesaurus_id, thesaurus_value in wanted_entities
        if filter_occurences(
            content,
            thesaurus_value,
            find_near_matches(thesaurus_value, content, max_l_dist=2),
        )
    )


def normalize_text(txt: str):
    """Apply simple normalization to make the comparison easier."""
    return txt.lower()


def extract_wanted_entities(
    chunk_contents: Iterable[str],
    complete_entity_sets: Iterable[list[CompleteEntity]],
    thesauri_factory: ThesaurusProvider,
) -> list[set[int] | None]:
    """Filter only the entities that fuzzymatch with wanted thesaurus.

    Arguments:
        chunk_contents: for each chunk, its text content
        complete_entity_sets: a set for each text chunk of occurring entities \
only in a group of entity types
        thesauri_factory: a set of wanted string values to be extracted in the \
same group of entity types

    ReturnType:
    A list for each text chunk of the matched thesaurus above the given distance treshold. If there is not any filtered entity for a given chunk, then None is returned for this chunk instead of the empty set.
    The empty set means that the chunk contains entities that match the group
    of entities of interests but these entities does not match the thesaurus.
    """
    load_and_normalized_thesauri = [
        (thesaurus_id, normalize_text(thesaurus_value))
        for thesaurus_id, thesaurus_value in thesauri_factory()
    ]
    return [
        extract_from_content(
            normalize_text(content), entity_set, load_and_normalized_thesauri
        )
        for content, entity_set in zip(
            chunk_contents, complete_entity_sets, strict=True
        )
    ]
