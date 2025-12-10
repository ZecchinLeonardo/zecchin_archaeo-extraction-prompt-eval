import os
import importlib
import pathlib

import pandas as pd
from pathlib import Path
# import traceback

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .pipeline_func import _as_list, _as_int_list, _as_str_list


class LoadScans(BaseEstimator, TransformerMixin):
    # def __init__(self, cache_csv: str | Path):
    #     self.cache_csv = Path(cache_csv)
    #     self._df = None
    # def fit(self, X, y=None):
    #     df = pd.read_csv(self.cache_csv)
    #     if "id" in df.columns:
    #         df["id"] = df["id"].astype("int64").astype(int)
    #     if "chunk_type" in df.columns:
    #         df["chunk_type"] = df["chunk_type"].apply(_as_str_list)
    #     if "chunk_page_position" in df.columns:
    #         df["chunk_page_position"] = df["chunk_page_position"].apply(_as_int_list)
    #     if "identified_thesaurus" in df.columns:
    #         df["identified_thesaurus"] = df["identified_thesaurus"].apply(_as_int_list)
    #     if "named_entities" in df.columns:
    #         df["named_entities"] = df["named_entities"].apply(_as_list)
    #     self._df = df
    #     return self
    # def transform(self, X):
    #     X = X.copy()
    #     if "id" in X.columns:
    #         X["id"] = X["id"].astype(int)
    #     return X.merge(self._df, on="id", how="inner")
    def __init__(self, cache_csv: str | Path):
        self.cache_csv = Path(cache_csv)
        self._df = None
    def fit(self, X, y=None):
        df = pd.read_csv(self.cache_csv)
        if "id" in df.columns:
            df["id"] = df["id"].astype("int64").astype(int)
        if "chunk_type" in df.columns:
            df["chunk_type"] = df["chunk_type"].apply(_as_str_list)
        if "chunk_page_position" in df.columns:
            df["chunk_page_position"] = df["chunk_page_position"].apply(_as_int_list)
        if "identified_thesaurus" in df.columns:
            df["identified_thesaurus"] = df["identified_thesaurus"].apply(_as_int_list)
        if "named_entities" in df.columns:
            df["named_entities"] = df["named_entities"].apply(_as_list)

        # Ensure embedding and text columns are non-null strings to satisfy schema checks
        for col in ("chunk_embedding_content", "chunk_content", "filename"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        self._df = df
        return self
    def transform(self, X):
        X = X.copy()
        if "id" in X.columns:
            X["id"] = X["id"].astype(int)
        return X.merge(self._df, on="id", how="inner")