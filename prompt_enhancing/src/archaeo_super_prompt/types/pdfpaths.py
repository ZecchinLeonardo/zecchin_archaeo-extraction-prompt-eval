import pandera.pandas as pa
import pandas as pd
from pathlib import Path
from typing import cast
from collections.abc import Iterable

from pandera.typing.pandas import DataFrame, Series

from .intervention_id import InterventionId


class PDFPathSchema(pa.DataFrameModel):
    id: Series[int]
    filepath: Series[str]


PDFPathDataset = DataFrame[PDFPathSchema]


def buildPdfPathDataset(
    items: Iterable[tuple[InterventionId, Path]],
) -> PDFPathDataset:
    ids, paths = cast(
        tuple[tuple[InterventionId, ...], tuple[Path, ...]],
        zip(*items, strict=True),
    )
    return PDFPathSchema.validate(
        pd.DataFrame({"id": ids, "filepath": [str(path) for path in paths]})
    )


def get_intervention_rows(ds: PDFPathDataset):
    return [
        (InterventionId(row["id"]), Path(row["filepath"]))
        for _, row in ds.iterrows()
    ]


def get_paths(ds: PDFPathDataset) -> list[Path]:
    return [Path(str_path) for str_path in cast(Series[str], ds["filepath"])]
