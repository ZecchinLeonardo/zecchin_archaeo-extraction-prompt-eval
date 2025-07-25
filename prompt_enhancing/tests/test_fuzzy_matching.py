"""Test the fuzzy matching algorithms."""

from archaeo_super_prompt.modeling.entity_extractor import fuzzy_match
from fuzzysearch import Match
from typing import NamedTuple

from archaeo_super_prompt.modeling.entity_extractor.types import CompleteEntity

SAMPLE_TEXT = """Il palazzo, infatti, si trova nel pieno centro cittadino a poche
decine di metri da Piazza dei Cavalieri e dal Lungarno Pacinotti5. L’edificio,
inoltre, è stato realizzato a partire dalla metà del XIV secolo per ospitare la
Piazza del Grano, sede del più importante mercato urbano, per poi essere
ampliato e ricostruito alla fine del XV secolo per fare posto alla nuova
Università di Pisa."""


def test_extended_expression_computation():
    """Test if the computation of extended match is correct."""

    class MatchWithExpectedExtendedMatch(NamedTuple):
        content: str
        match: Match
        expected_extended_expr: str

    test_set: list[MatchWithExpectedExtendedMatch] = [
        MatchWithExpectedExtendedMatch(
            "WE ARE IN PONTEDERA",
            Match(start=10, end=15, dist=0, matched="PONTE"),  # type: ignore
            "PONTEDERA",
        ),
        MatchWithExpectedExtendedMatch(
            "WE ARE IN PONTEDERA",
            Match(start=10, end=19, dist=0, matched="PONTEDERA"),  # type: ignore
            "PONTEDERA",
        ),
        MatchWithExpectedExtendedMatch(
            "WE ARE IN PONTEDERA",
            Match(start=0, end=2, dist=0, matched="WE"),  # type: ignore
            "WE",
        ),
        MatchWithExpectedExtendedMatch(
            "WE ARE IN PONTEDERA",
            Match(start=9, end=19, dist=0, matched=" PONTEDERA"),  # type: ignore
            " PONTEDERA",
        ),
        MatchWithExpectedExtendedMatch(
            "WE ARE IN AN APPARTEMENT",
            Match(start=15, end=19, dist=0, matched="PART"),  # type: ignore
            "APPARTEMENT",
        ),
        MatchWithExpectedExtendedMatch(
            "I am working for the Soprintendenza Archeologica della Toscana",
            Match(
                start=25,  # type: ignore
                end=57,  # type: ignore
                dist=2,  # type: ignore
                matched="intendenza Archeologica della To",  # type: ignore
            ),
            "Soprintendenza Archeologica della Toscana",
        ),
        # TODO: add a test for extending with computation
        # (eg. "sono in lungarno piacinotti. ", "lungarno piacinotti" -> "lungarno piacinotti")
    ]
    for content, match, expected_extended_match in test_set:
        assert (
            fuzzy_match.extended_expression(content, match)
            == expected_extended_match
        )


def test_text_normalization():
    """Test the normalization of values."""
    assert fuzzy_match.normalize_text("Piazza") == "piazza"
    assert fuzzy_match.normalize_text("Dei caValieri.") == "dei cavalieri."
    assert (
        fuzzy_match.normalize_text("lungarno pacinotti")
        == "lungarno pacinotti"
    )


def test_anti_partial_match_filter():
    """Test if all the partial matches which does not actually match with the thesaurus are filtered out."""
    content = fuzzy_match.normalize_text(SAMPLE_TEXT)
    inputs = [
        (
            "piazza dei cavalieri",
            [
                Match(
                    start=84,  # type: ignore
                    end=104,  # type: ignore
                    dist=0,  # type: ignore
                    matched="piazza dei cavalieri",  # type: ignore
                ),
            ],
        ),
        (
            "lungarno pacinotti",
            [
                Match(
                    start=111,  # type: ignore
                    end=129,  # type: ignore
                    dist=0,  # type: ignore
                    matched="lungarno pacinotti",  # type: ignore
                ),
            ],
        ),
        (
            "piazza del grano",
            [
                Match(start=224, end=240, dist=0, matched="piazza del grano"),  # type: ignore
            ],
        ),
        (
            "università di pisa",
            [
                Match(
                    start=370,  # type: ignore
                    end=388,  # type: ignore
                    dist=0,  # type: ignore
                    matched="università di pisa",  # type: ignore
                ),
            ],
        ),
        (
            "pisa",
            [
                Match(start=84, end=87, dist=1, matched="pia"),  # type: ignore
                Match(start=214, end=218, dist=1, matched="pita"),  # type: ignore
                Match(start=224, end=227, dist=1, matched="pia"),  # type: ignore
                Match(start=384, end=388, dist=0, matched="pisa"),  # type: ignore
            ],
        ),
        (
            "caval",
            [
                Match(start=95, end=100, dist=0, matched="caval"),  # type: ignore
            ],
        ),
    ]
    expected_answers: list[list[Match]] = [
        [Match(start=84, end=104, dist=0, matched="piazza dei cavalieri")],  # type: ignore
        [Match(start=111, end=129, dist=0, matched="lungarno pacinotti")],  # type: ignore
        [Match(start=224, end=240, dist=0, matched="piazza del grano")],  # type: ignore
        [Match(start=370, end=388, dist=0, matched="università di pisa")],  # type: ignore
        [Match(start=384, end=388, dist=0, matched="pisa")],  # type: ignore
        [],
    ]
    for (th_val, non_filtered_matches), filtered_matches in zip(
        inputs, expected_answers
    ):
        assert (
            fuzzy_match.filter_occurences(
                content, th_val, non_filtered_matches
            )
            == filtered_matches
        )


def test_fuzzy_matching_allowance():
    """Test if, in a noisy sample text, some thesaurus can still be found by the individual extractor."""
    normalized_thesauri = [
        (890, fuzzy_match.normalize_text("Lungarno Pacinotti"))
    ]
    result = fuzzy_match.extract_from_content(
        fuzzy_match.normalize_text(SAMPLE_TEXT),
        [
            CompleteEntity(
                entity="LUOGO", word="Lungarno Pacinotti", start=111, end=129
            )
        ],
        normalized_thesauri,
    )
    assert result is not None
    assert 890 in result


def test_main_extractor_spec():
    """Check if the main extractor return None or an empty set according to the input."""

    def thesauri():
        return [(890, "Lungarno Pacinotti")]

    result = fuzzy_match.extract_wanted_entities(
        iter(
            [
                SAMPLE_TEXT,
                SAMPLE_TEXT,
            ]
        ),
        iter(
            [
                [
                    CompleteEntity(
                        entity="LUOGO",
                        word="Lungarno Pacinotti",
                        start=111,
                        end=129,
                    )
                ],
                [],
            ]
        ),
        thesauri,
    )
    assert len(result) == 2
    assert result[0] is not None
    assert result[1] is None


def test_main_extractor():
    """Functional test of the main extractor."""

    def wanted_thesauri():
        return list(
            enumerate(
                [
                    "Piazza dei Cavalieri",
                    "Lungarno Pacinotti",
                    "Piazza del Grano",
                    "Università di Pisa",
                    "Pisa",
                    "Caval",
                ],
            )
        )

    chunks = iter(
        [
            SAMPLE_TEXT,
            SAMPLE_TEXT,
            "In questo testo non troverete alcun thesaurus.",
        ]
    )

    identified_entities_for_chunks = iter(
        [
            [
                CompleteEntity(
                    entity="LUOGO",
                    word="Lungarno Pacinotti",
                    start=111,
                    end=129,
                )
            ],
            [],
            [
                CompleteEntity(
                    entity="COGNOME",
                    word="thesaurus",
                    start=36,
                    end=45,
                )
            ],
        ]
    )
    results = fuzzy_match.extract_wanted_entities(
        chunks, identified_entities_for_chunks, wanted_thesauri
    )
    assert(results[0] == {0, 1, 2, 3, 4})
    assert(results[1] is None)
    assert(results[2] == set())
