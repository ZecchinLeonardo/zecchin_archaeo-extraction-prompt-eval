import pandas as pd
from pathlib import Path


def load_comune() -> list[str]:
    return pd.read_csv(
        Path(__file__).parent / "../../../data/raw/thesaurus/comune.csv"
    )["nome"].to_list()
