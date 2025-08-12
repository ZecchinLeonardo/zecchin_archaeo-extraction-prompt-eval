"""Loading of thesauri related to the comune and the province."""

from typing import NamedTuple
import pandas as pd
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import DataFrame, Index, Series

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


class ComuneData(DataFrameModel):
    """Data about a Comune."""

    comune_id: Index[int]
    name: Series[str]
    province_id: Series[int]


class ProvinciaData(DataFrameModel):
    """Data about a Province."""

    province_id: Index[int]
    name: Series[str]
    sigla: Series[str]  # 2-chars


def load_comune_with_provincie() -> tuple[
    DataFrame[ComuneData], DataFrame[ProvinciaData]
]:
    """Load the set of provincie thesaurus from an external reference table."""
    comune = pd.read_csv(_get_comune_file())
    province = pd.read_csv(_get_provincie_file(), keep_default_na=False)
    return ComuneData.validate(
        comune[comune["nome"].notnull() & comune["provincia"].notnull()][
            ["id_com", "nome", "provincia"]
        ]
        .rename(
            columns={
                "id_com": "comune_id",
                "nome": "name",
                "provincia": "province_id",
            }
        )
        .set_index("comune_id")
    ), ProvinciaData.validate(
        province[["id_prov", "nome", "sigla"]]
        .rename(columns={"id_prov": "province_id", "nome": "name"})
        .set_index("province_id")
    )
