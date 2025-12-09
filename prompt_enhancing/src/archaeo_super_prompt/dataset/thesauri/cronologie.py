"""Loader for chronology thesaurus (thesaurus_crono.csv).

This mirrors the structure used in `ritrovamenti.py` and uses
`get_cache_dir_for("raw", "thesaurus")` to locate the file.
"""
from typing import NamedTuple
import pandas as pd
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import DataFrame, Index, Series

from ...utils.cache import get_cache_dir_for


def _get_cronologia_file():
    return get_cache_dir_for("raw", "thesaurus") / "thesaurus_crono.csv"


def load_cronologia() -> pd.DataFrame:
    """Load the chronology thesaurus CSV and return it as a DataFrame.

    The file is expected to be delimited with `;`. The loader is permissive
    about column names and returns the raw DataFrame for downstream code to
    pick the appropriate column(s).
    """
    df = pd.read_csv(_get_cronologia_file(), sep=";", dtype=str, keep_default_na=False)
    return df


class CronologiaData(DataFrameModel):
    """Data about chronology thesaurus rows."""
    cronologia_id: Index[int]
    cronologia: Series[str]
