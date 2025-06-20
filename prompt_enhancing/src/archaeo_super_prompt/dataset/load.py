import pandas as pd
import numpy as np

from ..target_types import MagohData

from .postgresql_engine import get_entries
from .minio_engine import download_files
from ..cache import memory


@memory.cache
def _init_with_cache(size: int, seed: float, only_recent_entries=False):
    intervention_data, findings = get_entries(size, seed, only_recent_entries)
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
        ]
        if len(record) == 0:
            raise Exception(f"Unable to get record with id {id_}")
        record = record.iloc[0]
        dict_record = { }
        for k in record.keys():
            chunks = k.split(".")
            if len(chunks) != 2:
                continue
            prefix, suffix = chunks
            if prefix not in dict_record:
                dict_record[prefix] = {}
            value = record[k]
            if isinstance(value, np.bool):
                value = bool(value)
            if isinstance(value, np.int64):
                value = int(value)
            if isinstance(value, np.float64):
                value = float(value)
            dict_record[prefix][suffix] = value
        return MagohData(dict_record) #type: ignore

    @property
    def findings(self):
        return self._findings

    @property
    def files(self):
        return self._files
