from typing import Dict, Generator, Optional, Set, Tuple, cast, List
import requests
import fuzzywuzzy.process as fzwz

from .types import CompleteEntity, NerOutput, NerXXLEntities


def fetch_entities(chunks: List[str]):
    payload = {"chunks": chunks}
    response = requests.post("http://localhost:8884/ner", json=payload)
    response.raise_for_status()
    entities = list(
        map(
            lambda lst: list(map(lambda dct: NerOutput(**dct), lst)),
            cast(List[List[Dict]], response.json()),
        )
    )
    return entities


def postrocess_entities(
    entitiesPerTextChunk: List[List[NerOutput]], confidence_treshold: float
):
    """Return a set of the occured entities for each chunks
    Arguments:
    * confidence_treshold: a treshold between 0 and 1 to tolerate only a subset
    of entities
    """

    def gatherEntityChunks(entity_chunks: List[NerOutput]):
        entity_set: List[CompleteEntity] = list()
        current_accumulated_entity: Optional[CompleteEntity] = None
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
                    entity=cast(NerXXLEntities, current_entity_chunk.entity[2:]),
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
                and abs(current_entity_chunk.start - current_accumulated_entity.end)
                <= 1
            ):
                current_accumulated_entity.end = current_entity_chunk.end
                # Complete an entity with its additional chunks
                if current_entity_chunk.word.startswith("##"):
                    # the chunk belongs to the same entity word
                    current_accumulated_entity.word += current_entity_chunk.word[2:]
                else:
                    # the entity is composed of several words
                    current_accumulated_entity.word += " " + current_entity_chunk.word
        return entity_set

    return [gatherEntityChunks(entity_chunks) for entity_chunks in entitiesPerTextChunk]


def filter_entities(
    complete_entity_sets: List[Set[CompleteEntity]],
    allowed_entities: Set[NerXXLEntities],
) -> List[Set[CompleteEntity]]:
    """For each text chunk, keep only the entities included in the given group
    of allowed entity types.
    """
    return [
        set(filter(lambda e: e.entity in allowed_entities, s))
        for s in complete_entity_sets
    ]


def extract_wanted_entities(
    complete_entity_sets: List[Set[CompleteEntity]],
    wanted_entities: Set[str],
    distance_treshold: float,
) -> List[List[Tuple[CompleteEntity, List[str]]]]:
    """Filter only the entities that fuzzymatch with wanted thesaurus

    Arguments:
    * complete_entity_sets: a set for each text chunk of occurring entities
    only in a group of entity types
    * wanted_entities: a set of wanted string values to be extracted in the
    same group of entity types
    * distance_treshold: a float between 0 and 1

    ReturnType:
    A list for each text chunk of the matched entities. Each of them are paired
    with the thesaurus values they have matched above the given distance
    treshold.
    """

    def aux(complete_entity_set: Set[CompleteEntity]):
        return [
            (
                entity,
                [
                    matched_thesaurus
                    for matched_thesaurus, _ in cast(
                        Generator[Tuple[str, int]],
                        fzwz.extractWithoutOrder(
                            entity.word,
                            wanted_entities,
                            score_cutoff=int(distance_treshold * 100),
                        ),
                    )
                ],
            )
            for entity in complete_entity_set
        ]

    return [aux(ces) for ces in complete_entity_sets]
