"""Named Entities selector test."""

from typing import cast
from archaeo_super_prompt.modeling import entity_extractor
import pandas as pd

name_field = entity_extractor.NamedEntityField(
    name="people_name",
    compatible_entities={"COGNOME", "NOME"},
    thesaurus_values=lambda: list(
        enumerate(["DUPONT", "DOE", "John", "Jean"])
    ),
)
place_time_field = entity_extractor.NamedEntityField(
    name="luoghi",
    compatible_entities={"DATA", "LUOGO", "CODICE_POSTALE", "INDIRIZZO"},
    thesaurus_values=lambda: [
        (8, "giugno"),
        (5, "giugno"),
        (12, "Archeologia"),
        (10, "Convegno"),
    ],
)
chunks = [
    "Fino al 3 giugno potete iscriveri gratuitamente al Convegno",
    "joe dupont has met JEAN Doe il 3 luglio",
]
entities = [
    [
        entity_extractor.types.CompleteEntity(
            entity="DATA", word="giugno", start=10, end=16
        )
    ],
    [
        entity_extractor.types.CompleteEntity(
            entity="NOME", word="joe", start=0, end=2
        ),
        entity_extractor.types.CompleteEntity(
            entity="COGNOME", word="dupont", start=4, end=9
        ),
        entity_extractor.types.CompleteEntity(
            entity="DATA", word="luglio", start=50, end=56
        ),
    ],
]


def test_ne_selector():
    """Test if the pipeline is type-safe and returns something wanted."""
    input = entity_extractor.types.ChunksWithEntities.validate(
        pd.DataFrame(
            {
                "id": [455, 455],
                "filename": ["f1.pdf", "f2.pdf"],
                "chunk_type": [["table"], ["table"]],
                "chunk_page_position": [[1], [2]],
                "chunk_index": [0, 1],
                "chunk_embedding_content": chunks,
                "chunk_content": chunks,
                "named_entities": entities,
            }
        )
    )
    time_place_extractor = entity_extractor.NeSelector(
        to_extract=place_time_field
    )
    output = time_place_extractor.transform(input)
    identified_place_month = cast(
        list[list[int]], output["identified_thesaurus"].tolist()
    )[0]
    assert set(identified_place_month) == {8, 5, 10}


def test_ne_selector_names():
    """Test if the pipeline returns only fully-matching thesaurus."""
    input = entity_extractor.types.ChunksWithEntities.validate(
        pd.DataFrame(
            {
                "id": [455, 455],
                "filename": ["f1.pdf", "f2.pdf"],
                "chunk_type": [["table"], ["table"]],
                "chunk_page_position": [[1], [2]],
                "chunk_index": [0, 1],
                "chunk_embedding_content": chunks,
                "chunk_content": chunks,
                "named_entities": entities,
            }
        )
    )
    name_extractor = entity_extractor.NeSelector(to_extract=name_field)
    output = name_extractor.transform(input)
    identified_names = cast(
        list[list[int]], output["identified_thesaurus"].tolist()
    )[0]
    assert set(identified_names) == {0, 1, 3}
