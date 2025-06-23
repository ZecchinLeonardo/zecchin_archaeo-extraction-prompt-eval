from pathlib import Path
from typing import Iterable, List, NewType, Tuple, cast

from pandas import DataFrame, Series

from archaeo_super_prompt.types.intervention_id import InterventionId


PDFPathDataset = NewType("PDFPathDataset", DataFrame)


def buildPdfPathDataset(items: Iterable[Tuple[InterventionId, Path]]):
    if not items:
        return PDFPathDataset(DataFrame())
    ids, paths = cast(
        Tuple[Tuple[InterventionId, ...], Tuple[Path, ...]], zip(*items, strict=True)
    )
    return PDFPathDataset(
        DataFrame({"id": ids, "filepath": [str(path) for path in paths]})
    )


def get_intervention_rows(ds: PDFPathDataset):
    return (
        (InterventionId(row["id"]), Path(row["filepath"])) for _, row in ds.iterrows()
    )


def get_paths(ds: PDFPathDataset) -> List[Path]:
    return [Path(str_path) for str_path in cast(Series[str], ds["filepath"])]
