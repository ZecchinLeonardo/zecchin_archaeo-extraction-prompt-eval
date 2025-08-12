"""Uniform data results."""

import pandera.pandas as pa
from pandera.typing.pandas import Series


class ResultSchema(pa.DataFrameModel):
    """DataFrame with the result for one record and one field."""
    id: Series[int] = pa.Field()
    field_name: Series[str] = pa.Field()  # TODO: convert it into categories
    predicted_value: Series[pa.Object] = pa.Field(nullable=True)
    expected_value: Series[pa.Object] = pa.Field(nullable=True)
    evaluation_method: Series[str] = (
        pa.Field()
    )  # TODO: convert it into categories
    metric_value: Series[float] = pa.Field()
