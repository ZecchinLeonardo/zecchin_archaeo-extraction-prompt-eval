"""Utils for piping normalization functions."""

from collections.abc import Callable
from typing import Literal, NamedTuple
from pandera.pandas import DataFrameModel
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
    anno: Series[int]
    processed_date: Series[Date | None]


class InterventionDataForDateNormalizationRowSchema(NamedTuple):
    """Row schema of the class above."""

    idscheda: int
    data_protocollo: str
    data_intervento: str
    anno: int
    processed_date: Date | None


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
    current_answer = row.processed_date
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
                column_key=lambda df: df.apply(
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
            s.assign(processed_date=None), lazy=True
        ),
        functions,
    )
