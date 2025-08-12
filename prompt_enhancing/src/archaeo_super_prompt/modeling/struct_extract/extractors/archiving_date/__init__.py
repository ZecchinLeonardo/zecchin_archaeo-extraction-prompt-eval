"""Mock archiving extractor with the good data.

As a model for extracting this data is not done, for now
we assume this data as known and simulate this behaviour
with loading from the dataset.
"""

import datetime
from typing import Any, cast, override

import pandas as pd
from pandera.typing.pandas import DataFrame, Series

from .....dataset.load import MagohDataset
from .....types.intervention_id import InterventionId
from .....types.pdfchunks import PDFChunkDataset
from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ....types.detailed_evaluator import DetailedEvaluatorMixin


class ArchivingDateOutputSchema(BasePerInterventionFeatureSchema):
    """when indentifying the date of an intervention, we refer first to the date of protocol."""

    data_protocollo: datetime.date


class ArchivingDateProvider(DetailedEvaluatorMixin[Any, MagohDataset, Any]):
    """Give the answer of the ArchivingDate."""

    def __init__(self) -> None:
        """."""
        super().__init__()
        self._mds: MagohDataset | None = None

    @override
    def fit(self, X, y: MagohDataset, **kwargs):
        X = X  # unused
        kwargs = kwargs  # unused
        self._mds = y
        return self

    def filter_ids(self, y: MagohDataset, ids: set[InterventionId]):
        """Only keeps the records with an inserted archiving date."""
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(
                Series[bool], df["building__Data_Protocollo"].notnull()
            ),
        )

    @override
    def predict(
        self,
        X: PDFChunkDataset,
    ) -> DataFrame[ArchivingDateOutputSchema]:
        if self._mds is None:
            raise NotImplementedError(
                "Cannot infer the data of archiving. Please fit the model with the dataset so the answers can be output."
            )

        def to_date(dp: str):
            d, m, y = [int(k) for k in dp.strip().split("-")]
            return datetime.date(y, m, d)

        return ArchivingDateOutputSchema.validate(
            pd.DataFrame(
                [
                    {
                        "id": a.id,
                        "data_protocollo": to_date(
                            str(a.building__Data_Protocollo)
                        ),
                    }
                    for a in self._mds.get_answers(set(X["id"].to_list()))
                ]
            ).set_index("id")
        )

    @override
    def score(
        self, X, y, sample_weight=None
    ) -> float:
        X = X  # unused
        y = y  # unused
        sample_weight = sample_weight  # unused
        return 1.0

    @override
    def score_and_transform(self, X, y):
        return self.score(X, y), self.predict(X)
