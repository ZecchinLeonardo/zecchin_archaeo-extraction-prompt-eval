"""Code for loading thesaurus sets from data files."""

import pandas as pd

from ..utils.cache import get_cache_dir_for


def load_comune() -> list[tuple[int, str]]:
    """Load the thesarus values for the "Comune" field."""
    df = pd.read_csv(
        get_cache_dir_for("external", "thesaurus/comune.csv")
    )
    return list(
        (id_, nome) for _, id_, nome in df[["id", "nome"]].itertuples()
    )
