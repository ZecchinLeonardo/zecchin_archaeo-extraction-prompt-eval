"""Loading of thesauri related to the comune and the province."""

from typing import NamedTuple, cast
import pandas as pd

from ...utils.cache import get_cache_dir_for


def _get_comune_file():
    return get_cache_dir_for("raw", "thesaurus") / "comune.csv"


def _get_provincie_file():
    return get_cache_dir_for("raw", "thesaurus") / "provincie.csv"


def load_comune() -> list[tuple[int, str]]:
    """Load the thesarus values for the "Comune" field."""
    df = pd.read_csv(_get_comune_file())
    return list(
        (id_, nome)
        for _, id_, nome in df[["id", "nome"]][
            df["nome"].notnull()
        ].itertuples()
    )


class Provincia(NamedTuple):
    """Exhaustive data about a Province."""
    id_: int
    name: str
    sigla: str


class ComuneProvincia(NamedTuple):
    """Exhaustive data about a Comune."""
    comune: str  # the name and the id
    provincia: Provincia


def load_comune_with_provincie() -> dict[int, ComuneProvincia]:
    """Load the set of provincie thesaurus from an external reference table."""
    comune = pd.read_csv(_get_comune_file())
    provincie = pd.read_csv(_get_provincie_file(), keep_default_na=False)
    return {
        cast(int, row.id): ComuneProvincia(
            cast(str, row.nome_x),
            Provincia(
                cast(int, row.id_prov),
                cast(str, row.nome_y),
                cast(str, row.sigla),
            ),
        )
        for row in comune.merge(
            provincie, "right", left_on="provincia", right_on="id_prov"
        ).itertuples()
    }

