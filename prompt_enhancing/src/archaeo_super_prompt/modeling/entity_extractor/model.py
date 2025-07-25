"""Core functions for inferring and filtering named entities in chunks."""

import itertools
from typing import cast

import requests
from tqdm import tqdm

from .types import CompleteEntity, NerOutput, NerXXLEntities


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


def fetch_entities(chunks: list[str]):
    """Infer into the remote NER model to find named entities in each chunk."""
    return list(
        itertools.chain.from_iterable(
            _fetch_entities(list(c))
            for c in tqdm(
                itertools.batched(chunks, 50),
                desc="NER analysing",
                unit="Fraction of total text chunks",
                total=len(chunks) // 50 + int(len(chunks) % 50 != 0),
            )
        )
    )


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
