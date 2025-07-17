import pandas as pd
from tqdm import tqdm

from ..types.intervention_id import InterventionId
from ..types.pdfpaths import PDFPathSchema
from ..types.structured_data import (
    ExtractedStructuredDataSeries,
    structuredDataSchema,
    OutputStructuredDataSchema,
    outputStructuredDataSchema_itertuples
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
    """Class to interact with the general training/evaluation dataset.

    At the initialisation, fetch the data from the cache or from the remote
    dataset if needed.
    """

    def __init__(self, size: int, seed: float, only_recent_entries=False):
        """Fetch a maximum of `size` samples from the Magoh training database."""
        self._intervention_data, self._findings, self._files = (
            _init_with_cache(size, seed, only_recent_entries)
        )
        # TODO: only store the normalized intervention_data

    @property
    def intervention_data(self):
        return self._intervention_data

    @classmethod
    def _normalize_metadata_df(cls, df: pd.DataFrame):
        return OutputStructuredDataSchema.validate(
            df.filter(regex="^(scheda_intervento.id|(university|building).*)")
            .rename(columns={"scheda_intervento.id": "id"})
            .rename(columns=variabilize_column_name)
        )

    def get_answer(self, id_: InterventionId) -> ExtractedStructuredDataSeries:
        records = self._normalize_metadata_df(self._intervention_data)
        record = records[records["id"] == id_]
        if len(record) == 0:
            raise Exception(f"Unable to get record with id {id_}")
        return record.iloc[0].to_dict()

    def get_answers(self, ids: set[InterventionId]):
        """Return the answers for each of the asked interventions."""
        records = self._normalize_metadata_df(self._intervention_data)
        filtered = records[records["id"].isin(ids)]
        if len(filtered) != len(ids):
            raise Exception("All the asked interventions does not have their answers stored in the dataset")
        return outputStructuredDataSchema_itertuples(filtered)

    @property
    def findings(self):
        """Return a dataframe with the fetched findings data."""
        return self._findings

    def get_files_for_batch(self, ids: set[InterventionId]):
        """Return the files only realted to the given intervention ids."""
        return self._files[self._files["id"].isin(ids)]

    @property
    def files(self):
        """Return all the files with their related intervention id."""
        return self._files
