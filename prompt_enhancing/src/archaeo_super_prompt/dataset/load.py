import pandas as pd

from .postgresql_engine import get_entries
from .minio_engine import download_files
from ..cache import memory


@memory.cache
def __init_with_cache(size: int, seed: int):
    intervention_data, findings = get_entries(size, seed)
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
        ignore_index=True
    )
    return intervention_data, findings, files


class MagohDataset:
    def __init__(self, size: int, seed: int):
        """Fetch a maximum of `size` samples from the Magoh training database"""
        self._intervention_data, self._findings, self._files = __init_with_cache(
            size, seed
        )

    @property
    def intervention_data(self):
        return self._intervention_data

    @property
    def findings(self):
        return self._findings

    @property
    def files(self):
        return self._files

