from typing import Optional, Set, cast, List
import requests

from .types import CompleteEntity, NerOutput, NerXXLEntities


def fetch_entities(chunks: List[str]):
    payload = {"chunks": chunks}
    response = requests.post("http://localhost:8884/ner", json=payload)
    response.raise_for_status()
    entities = cast(List[List[NerOutput]], response.json())
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
        entity_set: Set[CompleteEntity] = set()
        current_entity: Optional[CompleteEntity] = None
        for entity_chunk in entity_chunks:
            # Edge-case when a chunks is under the confidence treshold
            # We only keep the already added confident chunk of the entity
            # and ignore the following chunks
            if entity_chunk.score < confidence_treshold:
                if current_entity is not None:
                    entity_set.add(current_entity)
                    current_entity = None
                continue

            if entity_chunk.entity.startswith("B-"):
                # Start a new entity with B- entities
                if current_entity is not None:
                    entity_set.add(current_entity)
                current_entity = CompleteEntity(
                    entity=cast(NerXXLEntities, entity_chunk.entity[2:]),
                    word=entity_chunk.word,
                    start=entity_chunk.start,
                    end=entity_chunk.end,
                )
            elif (
                current_entity is not None
                and entity_chunk.start == current_entity.end
                and entity_chunk.word.startswith("##")
            ):
                # Complete an entity with its additional chunks
                current_entity.word += entity_chunk.word[2:]
                current_entity.end = entity_chunk.end
        return entity_set

    return [gatherEntityChunks(entity_chunks) for entity_chunks in entitiesPerTextChunk]


def filter_entities(
    complete_entity_sets: List[Set[CompleteEntity]],
    allowed_entities: Set[NerXXLEntities],
) -> List[Set[CompleteEntity]]:
    return [
        set(filter(lambda e: e.entity in allowed_entities, s))
        for s in complete_entity_sets
    ]
