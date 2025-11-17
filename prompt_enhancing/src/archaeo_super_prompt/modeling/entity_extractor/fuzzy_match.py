"""Identification of thesaurus with fuzzymatching in text chunks."""

from collections.abc import Iterator

from fuzzysearch import find_near_matches, Match
from thefuzz import fuzz
import math

from .types import CompleteEntity, ThesaurusProvider

from ...utils import cache


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
    return content[extended_start:extended_end]


def filter_occurences(
    content: str, thesaurus_value: str, matches: list[Match]
) -> list[Match]:
    """Keep the matches whose extended expression still match with the thesarusus value.

    For example, if "PART" is detected in the content "WE ARE IN AN APPARTEMENT", then this match will be excluded.
    """

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


@cache.get_memory_for("interim").cache
def extract_from_content(
    content: str,
    entity_set: list[CompleteEntity],
    wanted_entities: list[tuple[int, str]],
) -> set[int] | None:
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


def normalize_text(txt: str):# -> str:
    # """Apply simple normalization to make the comparison easier."""
    # return txt.lower()
    """Normalize text for fuzzy matching; robust to NaN/None and non-string inputs."""
    if txt is None:
        return ""
    # numpy.nan and float('nan') are floats that satisfy math.isnan
    try:
        if isinstance(txt, float) and math.isnan(txt):
            return ""
    except Exception:
        pass
    if not isinstance(txt, str):
        try:
            txt = str(txt)
        except Exception:
            return ""
    return txt.strip().lower()


def extract_wanted_entities(
    chunk_contents: Iterator[str],
    complete_entity_sets: Iterator[list[CompleteEntity]],
    thesauri_factory: ThesaurusProvider,
) -> Iterator[set[int] | None]:
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
    # Robustly handle different shapes returned by thesauri_factory():
    # - iterable of (id, value)
    # - iterable of dicts
    # - pandas DataFrame/Series rows (which may be tuples, Series, or dict-like)
    raw_thesauri = list(thesauri_factory())
    load_and_normalized_thesauri: list[tuple[int | str, str]] = []
    for item in raw_thesauri:
        th_id = None
        th_val = None
        # Common: tuple/list with at least two elements
        if isinstance(item, (tuple, list)):
            if len(item) >= 2:
                th_id, th_val = item[0], item[1]
            else:
                # malformed entry, skip
                continue
        elif isinstance(item, dict):
            # try to pick a sensible id and value from the dict
            # prefer ('id','value') or ('comune_id','nome') patterns
            for key in ("id", "thesaurus_id", "comune_id"):  # possible id keys
                if key in item:
                    th_id = item[key]
                    break
            for key in ("value", "nome", "name", "iii_livello", "iii_lev"):
                if key in item:
                    th_val = item[key]
                    break
            # if still not found, fall back to first item
            if th_id is None or th_val is None:
                it = list(item.items())
                if it:
                    th_id = it[0][0] if th_id is None else th_id
                    th_val = it[0][1] if th_val is None else th_val
        else:
            # pandas Series/row-like or other object: try sequence then attributes
            try:
                # often a pandas row is indexable
                th_id, th_val = item[0], item[1]
            except Exception:
                # try attribute names
                th_id = getattr(item, "id", getattr(item, "comune_id", None))
                th_val = getattr(item, "nome", getattr(item, "name", None))

        if th_id is None or th_val is None:
            # skip malformed entries
            continue

        # normalize id to int when possible
        try:
            norm_id = int(th_id) if (isinstance(th_id, (int, str)) and str(th_id).isdigit()) else th_id
        except Exception:
            norm_id = th_id

        load_and_normalized_thesauri.append((norm_id, normalize_text(str(th_val))))

    return (
        extract_from_content(
            normalize_text(content), entity_set, load_and_normalized_thesauri
        )
        for content, entity_set in zip(chunk_contents, complete_entity_sets, strict=True)
    )
