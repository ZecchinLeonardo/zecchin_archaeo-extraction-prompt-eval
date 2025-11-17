"""Loading of thesauri related to the comune and the province."""

from typing import NamedTuple
import pandas as pd
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import DataFrame, Index, Series

from ...utils.cache import get_cache_dir_for


def _get_ritrovamenti_file():
    return get_cache_dir_for("raw", "thesaurus") / "thesaurus_iii_livello.csv"


def load_ritrovamento() -> DataFrame[DataFrameModel]:
    """Load the thesaurus third-level label and its code as a DataFrame.

    Reads the CSV using ';' as separator and returns a pandas DataFrame
    containing the columns `iii_lev` and `iii_lev_code`. Rows with null
    `iii_lev` are filtered out.
    """
    # the file uses ';' as delimiter
    df = pd.read_csv(_get_ritrovamenti_file(), sep=";")

    # Accept several possible column name variants used across sources
    possible_label_cols = ["iii_lev", "iii_livello", "university__iii_lev"]
    possible_code_cols = [
        "iii_lev_code",
        "iii_livello_id",
        "university__iii_lev_code",
        "iii_livello_code",
    ]

    label_col = next((c for c in possible_label_cols if c in df.columns), None)
    code_col = next((c for c in possible_code_cols if c in df.columns), None)

    if label_col is None or code_col is None:
        raise ValueError(
            f"expected one of label cols {possible_label_cols} and one of code cols {possible_code_cols} in thesaurus file; found {df.columns.tolist()}"
        )

    # Keep only relevant columns and drop null labels, then normalize names
    out = df[[label_col, code_col]][df[label_col].notnull()].reset_index(drop=True)
    out = out.rename(columns={label_col: "iii_lev", code_col: "iii_lev_code"})

    return out



class RitrovamentiData(DataFrameModel):
    """Data about a Ritrovamenti."""
    ritrovamento_id: Index[int]
    ritrovamento: Series[str]


# def load_comune_with_provincie() -> tuple[
#     DataFrame[ComuneData], DataFrame[ProvinciaData]
# ]:
#     """Load the set of provincie thesaurus from an external reference table."""
#     comune = pd.read_csv(_get_comune_file())
#     province = pd.read_csv(_get_provincie_file(), keep_default_na=False)
#     return ComuneData.validate(
#         comune[comune["nome"].notnull() & comune["provincia"].notnull()][
#             ["id_com", "nome", "provincia"]
#         ]
#         .rename(
#             columns={
#                 "id_com": "comune_id",
#                 "nome": "name",
#                 "provincia": "province_id",
#             }
#         )
#         .set_index("comune_id")
#     ), ProvinciaData.validate(
#         province[["id_prov", "nome", "sigla"]]
#         .rename(columns={"id_prov": "province_id", "nome": "name"})
#         .set_index("province_id")
#     )
