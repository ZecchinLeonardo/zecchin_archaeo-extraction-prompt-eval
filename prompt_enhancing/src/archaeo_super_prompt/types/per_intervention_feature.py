"""Global base class for per-intervention extracted features dataframe schemas."""

from pandera.pandas import DataFrameModel
from pandera.typing.pandas import Index


class BasePerInterventionFeatureSchema(DataFrameModel):
    """Base schema for dataframes containing extracted features related to one intervention."""
    id: Index[int]
