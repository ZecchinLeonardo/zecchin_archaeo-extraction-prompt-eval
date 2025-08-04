"""Utils for piping normalization functions."""

from collections.abc import Callable
from typing import Literal, NamedTuple, Optional
import pandas as pd
from pandera.pandas import DataFrameModel, Field
from pandera.typing.pandas import DataFrame, Series

Precision = Literal["day", "month", "year"]

class Date(NamedTuple):
    """Not completely normalized date, but the day, the month and the year are already separated by /."""

    start_date: str # d/m/y
    end_date: str # d/m/y
    precision: Precision


class RawInterventionDataForDateNormalization(DataFrameModel):
    """This is the schema of usefull columns for normalizing the intervention dates."""

    idscheda: Series[int]
    data_protocollo: Series[str]
    data_intervento: Series[str]
    anno: Series[int]


class InterventionDataForDateNormalization(DataFrameModel):
    """This is the schema of usefull columns for normalizing the intervention dates."""

    idscheda: Series[int]
    data_protocollo: Series[str]
    data_intervento: Series[str]
    anno: Series[pd.Int32Dtype]
    norm_date: Optional[Date] = Field(nullable=True)  # noqa: UP045


class InterventionDataForDateNormalizationRowSchema(NamedTuple):
    """Row schema of the class above."""

    idscheda: int
    data_protocollo: str
    data_intervento: str
    anno: int
    norm_date: Date | None


DateProcessor = Callable[
    [InterventionDataForDateNormalizationRowSchema], Date | None
]


def process_if_not_yet(
    row: InterventionDataForDateNormalizationRowSchema, fn: DateProcessor
) -> Date | None:
    """For each row not processed yet, apply a normalization function.

    This normalization function try to normalize if the humanly-input date
    matches with patterns that it supports. Else, it returns None.
    """
    current_answer = row.norm_date
    if current_answer is not None:
        return current_answer
    return fn(row)


def pipe(
    s: DataFrame[RawInterventionDataForDateNormalization],
    functions: tuple[DateProcessor, ...],
) -> DataFrame[InterventionDataForDateNormalization]:
    """Apply to the raw date df a range of normalization functions.

    This functions tries to cover a maximum of humanly-input dates.
    """

    def pipe_aux(
        s: DataFrame[InterventionDataForDateNormalization],
        functions: tuple[DateProcessor, ...],
    ):
        if not functions:
            return s
        return pipe_aux(
            s.assign(
                norm_date=lambda df: df.apply(
                    lambda row: process_if_not_yet(
                        InterventionDataForDateNormalizationRowSchema(
                            *tuple(row)
                        ),
                        functions[0],
                    ),
                    axis=1,
                )
            ),
            (*functions[1:],),
        )

    return pipe_aux(
        InterventionDataForDateNormalization.validate(
            s.assign(norm_date=None), lazy=True
        ),
        functions,
    )
