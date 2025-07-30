"""Module gathering the loaders for a training/evaluation dataset.

The data are slightly-transformed records of the Magoh database.
"""

from collections.abc import Callable
from typing import NamedTuple
import pandas as pd
from pandera.typing.pandas import DataFrame, Series
from tqdm import tqdm

from ..types.intervention_id import InterventionId
from ..types.pdfpaths import PDFPathSchema
from ..types.structured_data import (
    ExtractedStructuredDataSeries,
    structuredDataSchema,
    OutputStructuredDataSchema,
    outputStructuredDataSchema_itertuples,
)

from .postgresql_engine import get_entries, get_entries_with_ids
from .minio_engine import download_files
from ..utils.cache import get_memory_for
from ..utils.norm import variabilize_column_name


def _parse_intervention_data(intervention_data__df: pd.DataFrame):
    filtered_df = intervention_data__df.filter(
        regex="^(scheda_intervento.id|(university|building|check).*)"
    ).astype(
        {
            "university.Numero di saggi": "UInt32",
            "university.Geologico": "boolean",
            "university.Profondità falda": "Float64",
            "university.Profondità massima": "Float64",
        }
    )

    return structuredDataSchema.validate(filtered_df)


def _parse_and_get_files(intervention_data: pd.DataFrame):
    intervention_data = _parse_intervention_data(intervention_data)
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
    return intervention_data, files


@get_memory_for("external").cache
def _init_with_cache(size: int, seed: float, only_recent_entries=False):
    intervention_data, findings = get_entries(size, seed, only_recent_entries)
    intervention_data, files = _parse_and_get_files(intervention_data)
    return intervention_data, findings, files


@get_memory_for("external").cache
def _init_with_cache_for_ids(ids: set[int]):
    intervention_data, findings = get_entries_with_ids(ids)
    intervention_data, files = _parse_and_get_files(intervention_data)
    return intervention_data, findings, files


class SamplingParams(NamedTuple):
    """Parametres for sampling records in the training dataset."""
    size: int
    seed: float
    only_recent_entries: bool


"""A set of interventions identified by their schedaid."""
type IdSet = set[int]


class MagohDataset:
    """Class to interact with the general training/evaluation dataset.

    At the initialisation, fetch the data from the cache or from the remote
    dataset if needed.
    """

    def __init__(self, params: IdSet | SamplingParams):
        """Fetch intervention records from the Magoh's training database.

        Args:
            params: a set of intervention identifiers to be fetched or a group \
of sampling params to randomly fetch intervention records 
        """
        if isinstance(params, SamplingParams):
            size, seed, only_recent_entries = params
            intervention_data, self._findings, self._files = _init_with_cache(
                size, seed, only_recent_entries
            )
        else:
            intervention_data, self._findings, self._files = (
                _init_with_cache_for_ids(params)
            )
        self._intervention_data = self._normalize_metadata_df(
            intervention_data
        )

    @property
    def intervention_data(self):
        """A DataFrame with the truth metadata of registered records in Magoh."""
        return self._intervention_data

    @classmethod
    def _normalize_metadata_df(cls, df: pd.DataFrame):
        return OutputStructuredDataSchema.validate(
            df.filter(regex="^(scheda_intervento.id|(university|building).*)")
            .rename(columns={"scheda_intervento.id": "id"})
            .rename(columns=variabilize_column_name)
        )

    def get_answer(self, id_: InterventionId) -> ExtractedStructuredDataSeries:
        """Return the metadata of a magoh record with the given id."""
        records = self._intervention_data
        record = records[records["id"] == id_]
        if len(record) == 0:
            raise Exception(f"Unable to get record with id {id_}")
        return record.iloc[0].to_dict()

    def filter_good_records_for_training(
        self,
        ids: set[InterventionId],
        condition: Callable[
            [DataFrame[OutputStructuredDataSchema]], Series[bool]
        ],
    ) -> set[InterventionId]:
        """Return only the ids for which the intervention records match a given condition.

        Args:
            ids: the set of interventions to select
            condition: a function taking the training metadata dataframe and \
returning a series of boolean to filter the records with unusable values 
        """
        only_ids = self._intervention_data[
            self._intervention_data["id"].isin(ids)
        ]
        return set(
            InterventionId(id_)
            for id_ in only_ids[condition(only_ids)]["id"].to_list()
        )

    def get_answers(self, ids: set[InterventionId]):
        """Return the answers for each of the asked interventions."""
        records = self._intervention_data
        filtered = records[records["id"].isin(ids)]
        if len(filtered) != len(ids):
            raise Exception(
                "All the asked interventions does not have their answers stored in the dataset"
            )
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
