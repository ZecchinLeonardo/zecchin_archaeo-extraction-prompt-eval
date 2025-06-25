from typing import cast
import pandas as pd

from ..target_types import MagohData
from ..types.structured_data import structuredDataSchema

from .postgresql_engine import get_entries
from .minio_engine import download_files
from ..cache import memory


def parse_intervention_data(intervention_data__df: pd.DataFrame):
    filtered_df = intervention_data__df.filter(
        regex="^(scheda_intervento.id|(university|building|check).*)"
    )
    filtered_df["university.Numero di saggi"] = filtered_df[
        "university.Numero di saggi"
    ].astype('UInt32')
    filtered_df["university.Geologico"] = filtered_df["university.Geologico"].astype(
        "boolean"
    )

    return structuredDataSchema.validate(filtered_df)


@memory.cache
def _init_with_cache(size: int, seed: float, only_recent_entries=False):
    intervention_data, findings = get_entries(size, seed, only_recent_entries)
    intervention_data = parse_intervention_data(intervention_data)
    files = pd.concat(
        [
            pd.DataFrame(
                [
                    {"scheda_intervento.id": id_, "filepath": path}
                    for path in download_files(id_)
                ]
            )
            for id_ in intervention_data["scheda_intervento.id"]
        ],
        ignore_index=True,
    )
    return intervention_data, findings, files


class MagohDataset:
    def __init__(self, size: int, seed: float, only_recent_entries=False):
        """Fetch a maximum of `size` samples from the Magoh training database"""
        self._intervention_data, self._findings, self._files = _init_with_cache(
            size, seed, only_recent_entries
        )

    @property
    def intervention_data(self):
        return self._intervention_data

    def get_answer(self, id_: int) -> MagohData:
        record = self._intervention_data[
            self._intervention_data["scheda_intervento.id"] == id_
        ].filter(regex="^(scheda_intervento.id|(university|building).*)")
        if len(record) == 0:
            raise Exception(f"Unable to get record with id {id_}")
        record = record.iloc[0]
        dict_record = {}
        for k in record.keys():
            chunks = k.split(".")
            if len(chunks) != 2:
                continue
            prefix, suffix = chunks
            if prefix not in dict_record:
                dict_record[prefix] = {}
            value = record[k]
            dict_record[prefix][suffix] = value
        return MagohData(cast(MagohData, dict_record))

    @property
    def findings(self):
        return self._findings

    @property
    def files(self):
        return self._files
