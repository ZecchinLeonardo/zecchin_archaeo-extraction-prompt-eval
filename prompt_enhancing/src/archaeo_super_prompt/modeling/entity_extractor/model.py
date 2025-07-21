"""Core functions for inferring and filtering named entities in chunks."""

from logging import warning
from typing import cast
from collections.abc import Generator
import requests
import functools as fnt
import thefuzz.process as fzwz_p
import thefuzz.fuzz as fzwz
import re

from archaeo_super_prompt.types.thesaurus import ThesaurusProvider

from .types import CompleteEntity, NerOutput, NerXXLEntities
from ...utils import cache


def _fetch_entities(chunks: list[str]) -> list[list[NerOutput]]:
    if not chunks:
        return []
    print("Fetching the transformers model")
    payload = {"chunks": chunks}
    response = requests.post("http://localhost:8884/ner", json=payload)
    response.raise_for_status()
    entities = list(
        map(
            lambda lst: list(map(lambda dct: NerOutput(**dct), lst)),
            cast(list[list[dict]], response.json()),
        )
    )
    return entities


def cache_remote_ner_results(inpt, output=None):
    """Dummy identity function for caching the fetched NER resuts."""
    return cache.identity_function(inpt, output)


def fetch_entities(chunks: list[str]):
    """Infer into the remote NER model to find named entities in each chunk."""
    # TODO: correct these iter-list-iter spaghetti
    return [
        ner_result
        for _, ner_result in cache.escape_expensive_run_when_cached(
            cache_remote_ner_results,
            cache.get_memory_for("interim"),
            lambda txt: txt,
            lambda cit: iter(_fetch_entities(list(cit))),
            iter(chunks),
        )
    ]


def postrocess_entities(
    entitiesPerTextChunk: list[list[NerOutput]], confidence_treshold: float
):
    """Return a set of the occured entities for each chunks.

    Arguments:
        entitiesPerTextChunk: for each chunk, a list of its retrieved \
entities ordered by their occurence in the chunk's text content
        confidence_treshold: a treshold between 0 and 1 to tolerate only a \
subset of entities
    """

    def gatherEntityChunks(entity_chunks: list[NerOutput]):
        entity_set: list[CompleteEntity] = list()
        current_accumulated_entity: CompleteEntity | None = None
        for current_entity_chunk in entity_chunks:
            # Edge-case when a chunks is under the confidence treshold
            # We only keep the already added confident chunk of the entity
            # and ignore the following chunks
            if current_entity_chunk.score < confidence_treshold:
                if current_accumulated_entity is not None:
                    entity_set.append(current_accumulated_entity)
                    current_accumulated_entity = None
                continue

            if current_entity_chunk.entity.startswith("B-"):
                # Start a new entity with B- entities
                if current_accumulated_entity is not None:
                    entity_set.append(current_accumulated_entity)
                current_accumulated_entity = CompleteEntity(
                    entity=cast(
                        NerXXLEntities, current_entity_chunk.entity[2:]
                    ),
                    word=current_entity_chunk.word,
                    start=current_entity_chunk.start,
                    end=current_entity_chunk.end,
                )
            elif (
                current_accumulated_entity is not None
                # the condition below allows entities of the same type that
                # are consecutive or separated by one space to be merged
                # WARN: it is expected that the output content of the ner model
                # is normalized so words are only separated by 1 space at
                # maximum
                and abs(
                    current_entity_chunk.start - current_accumulated_entity.end
                )
                <= 1
            ):
                current_accumulated_entity.end = current_entity_chunk.end
                # Complete an entity with its additional chunks
                if current_entity_chunk.word.startswith("##"):
                    # the chunk belongs to the same entity word
                    current_accumulated_entity.word += (
                        current_entity_chunk.word[2:]
                    )
                else:
                    # the entity is composed of several words
                    current_accumulated_entity.word += (
                        " " + current_entity_chunk.word
                    )
        return entity_set

    return [
        gatherEntityChunks(entity_chunks)
        for entity_chunks in entitiesPerTextChunk
    ]


def filter_entities(
    complete_entity_sets: list[
        list[CompleteEntity]
    ],  # List[Set[CompleteEntity]]
    allowed_entities: set[NerXXLEntities],
) -> list[list[CompleteEntity]]:  # List[Set[CompleteEntity]]
    """For each text chunk, keep only the entities included in the given group of allowed entity types."""
    return [
        list(filter(lambda e: e.entity in allowed_entities, s))
        for s in complete_entity_sets
    ]


def extract_wanted_entities(
    chunk_contents: list[str],
    complete_entity_sets: list[list[CompleteEntity]],
    wanted_entities: ThesaurusProvider,
    distance_treshold: float,
) -> list[set[int] | None]:
    """Filter only the entities that fuzzymatch with wanted thesaurus.

    Arguments:
        chunk_contents: for each chunk, its text content
        complete_entity_sets: a set for each text chunk of occurring entities \
only in a group of entity types
        wanted_entities: a set of wanted string values to be extracted in the \
same group of entity types
        distance_treshold: a float between 0 and 1

    ReturnType:
    A list for each text chunk of the matched thesaurus above the given distance treshold. If there is not any filtered entity for a given chunk, then None is returned for this chunk instead of the empty set.
    The empty set means that the chunk contains entities that match the group
    of entities of interests but these entities does not match the thesaurus.
    """
    if distance_treshold > 0.75:
        warning("The selector might be very untolerant.")

    def aux(chunk_content: str, complete_entity_set: list[CompleteEntity]):
        """Auxiliary function for processing the entities of one chunk.

        To be mapped into an iterable.

        Arguments:
            chunk_content: the whole chunk's text content
            complete_entity_set: a not empty list of entities identified in \
the chunk
        """

        def normalize_text(txt: str):
            return txt.strip().lower()

        def anti_small_words_filter(txt: str):
            """Filter to not include wrong small words in partial ratio selection.

            Expect the text to be already normalized. Augment the size of
            spaces between words so the levenstein distance between a small
            wanted word and expressions which can have this small word as
            prefix is higher.
            """
            MAX_SIZE = 50

            def repeat_until(word: str, n: int):
                return (word * ((n + len(word) - 1) // len(word)))[:n]

            result = re.sub(
                r"\b(\w+)\b", lambda m: repeat_until(m.group(1), MAX_SIZE), txt
            )
            return result

        def pre_tranform(txt: str):
            r = anti_small_words_filter(normalize_text(txt))
            return r

        wanted = {k: pre_tranform(v) for k, v in wanted_entities()}

        def extract(content: str):
            return [
                matched_thesaurus_id
                for _, _, matched_thesaurus_id in cast(
                    Generator[tuple[str, int, int]],
                    fzwz_p.extractWithoutOrder(
                        pre_tranform(content),
                        wanted,
                        scorer=fzwz.partial_ratio,
                        score_cutoff=int(distance_treshold*100),
                    ),
                )
            ]

        matches_from_content = extract(chunk_content)
        matches_from_entities = fnt.reduce(
            lambda thesaurus_set,
            new_extracted_thesaurus_group: thesaurus_set.union(
                new_extracted_thesaurus_group
            ),
            [extract(entity.word) for entity in complete_entity_set],
            cast(set[int], set()),
        )
        return matches_from_entities.union(matches_from_content)

    return [
        aux(chunk_content, ces) if ces else None
        for chunk_content, ces in zip(
            chunk_contents, complete_entity_sets, strict=True
        )
    ]
