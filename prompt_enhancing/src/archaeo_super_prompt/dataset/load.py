from typing import Set
import pandas as pd
from tqdm import tqdm

from ..types.intervention_id import InterventionId
from ..types.pdfpaths import PDFPathSchema
from ..types.structured_data import (
    ExtractedStructuredDataSeries,
    structuredDataSchema,
    outputStructuredDataSchema,
)

from .postgresql_engine import get_entries
from .minio_engine import download_files
from ..utils.cache import get_memory_for
from ..utils.norm import variabilize_column_name


def parse_intervention_data(intervention_data__df: pd.DataFrame):
    filtered_df = intervention_data__df.filter(
        regex="^(scheda_intervento.id|(university|building|check).*)"
    ).astype(
        {
            "university.Numero di saggi": "UInt32",
            "university.Geologico": "boolean",
        }
    )

    return structuredDataSchema.validate(filtered_df)


@get_memory_for("external").cache
def _init_with_cache(size: int, seed: float, only_recent_entries=False):
    intervention_data, findings = get_entries(size, seed, only_recent_entries)
    intervention_data = parse_intervention_data(intervention_data)
    files = PDFPathSchema.validate(
        pd.concat(
            [
                pd.DataFrame(
                    [
                        {"id": id_, "filepath": str(path.resolve())}
                        for path in download_files(id_)
                    ]
                )
                for id_ in tqdm(
                    intervention_data["scheda_intervento.id"],
                    desc="Downloaded files",
                    unit="interventions",
                )
            ],
            ignore_index=True,
        )
    )
    return intervention_data, findings, files


class MagohDataset:
    def __init__(self, size: int, seed: float, only_recent_entries=False):
        """Fetch a maximum of `size` samples from the Magoh training database"""
        self._intervention_data, self._findings, self._files = (
            _init_with_cache(size, seed, only_recent_entries)
        )

    @property
    def intervention_data(self):
        return self._intervention_data

    def get_answer(self, id_: InterventionId) -> ExtractedStructuredDataSeries:
        record = outputStructuredDataSchema.validate(
            self._intervention_data[
                self._intervention_data["scheda_intervento.id"] == id_
            ]
            .filter(regex="^(scheda_intervento.id|(university|building).*)")
            .rename(columns={"scheda_intervento.id": "id"})
            .rename(columns=variabilize_column_name)
        )
        if len(record) == 0:
            raise Exception(f"Unable to get record with id {id_}")
        return record.iloc[0].to_dict()

    @property
    def findings(self):
        return self._findings

    def get_files_for_batch(self, ids: Set[InterventionId]):
        return self._files[self._files["id"].isin(ids)]

    @property
    def files(self):
        return self._files
