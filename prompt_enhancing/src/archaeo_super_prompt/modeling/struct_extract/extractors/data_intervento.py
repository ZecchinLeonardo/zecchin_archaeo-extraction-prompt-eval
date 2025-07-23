"""LLM-based extraction of the Date of Intervention.

We expect the model to predict a window of date for the start of the
intervention. This model has a known/guessed date of archiving of the report
and can output a window at least before this date. The precision of the window
is among those below :
1. Day
2. Month
3. Year
"""

import datetime
from typing import Literal, TypedDict, cast, override

import dspy
import pandas as pd
import pandera.pandas as pa
import pydantic
from pandera.typing.pandas import Series
import re

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.types import BaseKnowledgeDataScheme
from archaeo_super_prompt.types.intervention_id import InterventionId

from ....types.per_intervention_feature import BasePerInterventionFeatureSchema
from ..field_extractor import FieldExtractor, TypedDspyModule

# -- DSPy part

type Month = Literal[
    "Gennaio",
    "Febbraio",
    "Marzo",
    "Aprile",
    "Maggio",
    "Giugno",
    "Luglio",
    "Agosto",
    "Settembre",
    "Ottobre",
    "Novembre",
    "Dicembre",
]

MONTHS: list[Month] = [
    "Gennaio",
    "Febbraio",
    "Marzo",
    "Aprile",
    "Maggio",
    "Giugno",
    "Luglio",
    "Agosto",
    "Settembre",
    "Ottobre",
    "Novembre",
    "Dicembre",
]


class Data(pydantic.BaseModel):
    """Un data. A volte, il giorno o il mese possono avere un valore artificiale quando la precisione non consente di prevedere questi campi."""

    giorno: int
    mese: Month
    anno: int


class StimareDataDellIntervento(dspy.Signature):
    """Degli framenti datti di relazione archeologiche, stima il momento di partenza dell'indagine in una finestra de due date, con un precisione al giorno, al mese o all'anno più vicino. Se non ci sono molte informazione, ritorna almeno una finestra prima di la data di archiviazone datta.

    1. Innanzitutto, determina la precisione con cui puoi approssimare la finestra.
    2. Quindi, determina la finestra, inserendo valori predefiniti (ma ben tipizzati) nei campi non coperti dalla precisione.
       a. Se possibile, restringi la finestra a un punto impostando le stesse date minima e massima.
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    data_di_archiviazone: Data = dspy.InputField()

    data_minima_di_inizio: Data = dspy.OutputField()
    data_massima_di_inizio: Data = dspy.OutputField()
    precisione: Literal["giorno", "mese", "anno"] = dspy.OutputField()


class DataInterventoInputData(TypedDict):
    """Chunks of reports of an archaeological intervention with supposed information about the date of the intervention."""

    fragmenti_relazione: str
    data_di_archiviazone: Data


class DataInterventoOutputData(TypedDict):
    """A predicted maximum date for the intervention, in un window."""

    start_day: int
    start_month: int  # between 1 and 12
    start_year: int
    end_day: int
    end_month: int  # between 1 and 12
    end_year: int
    precision: Literal["day", "month", "year"]


class EstimateInterventionDate(
    TypedDspyModule[DataInterventoInputData, DataInterventoOutputData]
):
    """DSPy model for the extraction of the date of the intervention."""

    def __init__(self, callbacks=None):
        """Initialize only a chain of thought."""
        super().__init__(callbacks)
        self._estrattore_della_data = dspy.ChainOfThought(
            StimareDataDellIntervento
        )

    def forward(self, fragmenti_relazione: str, data_di_archiviazone: Data):
        """Simple date parsing."""
        result = cast(
            dspy.Prediction,
            self._estrattore_della_data(
                fragmenti_relazione=fragmenti_relazione,
                data_di_archiviazone=data_di_archiviazone,
            ),
        )

        DEFAULT_WRONG_DATE = Data(
            giorno=25, mese="Dicembre", anno=-1
        )  # The child was born
        TO_ENGLISH_PRECISION: dict[
            Literal["giorno", "mese", "anno"], Literal["day", "month", "year"]
        ] = {"giorno": "day", "mese": "month", "anno": "year"}

        data_minima_di_inizio = cast(
            Data, result.get("data_minima_di_inizio", DEFAULT_WRONG_DATE)
        )
        data_massima_di_inizio = cast(
            Data, result.get("data_minima_di_inizio", DEFAULT_WRONG_DATE)
        )
        precisione = cast(
            Literal["giorno", "mese", "anno"], result.get("precisione", "day")
        )

        return dspy.Prediction(
            **DataInterventoOutputData(
                start_day=data_minima_di_inizio.giorno,
                start_month=MONTHS.index(data_minima_di_inizio.mese),
                start_year=data_minima_di_inizio.anno,
                end_day=data_massima_di_inizio.giorno,
                end_month=MONTHS.index(data_massima_di_inizio.mese),
                end_year=data_massima_di_inizio.anno,
                precision=TO_ENGLISH_PRECISION[precisione],
            )
        )


# -- SKlearn part


class ArchivingDateKnowledge(BaseKnowledgeDataScheme):
    """The already inferred date of archiving."""

    data_protocollo: datetime.date


class DateFeatSchema(BasePerInterventionFeatureSchema):
    """Extracted data about the intervention start date."""

    intervention_date_start: Series[pa.DateTime]
    intervention_date_end: Series[pa.DateTime]
    intervention_date_precision: Literal["day", "month", "year"] = pa.Field(
        isin=["day", "month", "year"]
    )


class ComuneExtractor(
    FieldExtractor[
        DataInterventoInputData,
        DataInterventoOutputData,
        ArchivingDateKnowledge,
        DateFeatSchema,
    ]
):
    """Dspy-LLM-based extractor of the comune data."""

    def __init__(self, llm_model: dspy.LM) -> None:
        """Initialize the extractor with providing it the llm which will be used."""
        example = (
            DataInterventoInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
Lo scavo è iniziato il 18 marzo 1985 ed è terminato il 20 marzo.""",
                data_di_archiviazone=Data(giorno=5, mese="Maggio", anno=1985),
            ),
            DataInterventoOutputData(
                start_day=18,
                start_month=3,
                start_year=1985,
                end_day=18,
                end_month=3,
                end_year=1985,
                precision="day",
            ),
        )
        super().__init__(
            llm_model,
            EstimateInterventionDate(),
            example,
        )

    @override
    def _to_dspy_input(self, x) -> DataInterventoInputData:
        date_of_archiving = x.knowledge["data_protocollo"]
        return DataInterventoInputData(
            fragmenti_relazione=x.merged_chunks,
            data_di_archiviazone=Data(
                giorno=date_of_archiving.day,
                mese=MONTHS[date_of_archiving.month],
                anno=date_of_archiving.year,
            ),
        )

    @override
    def _transform_dspy_output(self, y):
        ids, values = cast(
            tuple[
                tuple[InterventionId, ...],
                tuple[DataInterventoOutputData, ...],
            ],
            zip(*y.items()),
        )
        return DateFeatSchema.validate(
            pd.DataFrame(
                {
                    "id": [int(id_) for id_ in ids],
                    "intervention_date_start": pd.to_datetime(
                        [
                            f"{y['start_year']}-{y['start_month']}-{y['start_day']}"
                            for y in values
                        ],
                        format="%Y-%m-%d",
                    ),
                    "intervention_date_end": pd.to_datetime(
                        [
                            f"{y['end_year']}-{y['end_month']}-{y['end_day']}"
                            for y in values
                        ],
                        format="%Y-%m-%d",
                    ),
                    "intervention_date_precision": [
                        y["precision"] for y in values
                    ],
                }
            ),
            # TODO: add this argument
            # lazy=True,
        )

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        if predicted == expected:
            return 1, TRESHOLD
        return int(predicted == expected), TRESHOLD
        # TODO: return a real evaluation as below
        # if (
        #     predicted["month"] == expected["month"]
        #     and predicted["year"] == expected["year"]
        # ):
        #     if expected["day"] is None:
        #         return 0.9, TRESHOLD
        #     if predicted["day"] is None:
        #         return 0.7, TRESHOLD
        #     return 0.7 if predicted["day"] > expected["day"] else 0.6, TRESHOLD
        # if expected["day"] is None:
        #     delay = datetime(
        #         predicted["year"], predicted["month"], 1
        #     ) - datetime(expected["year"], expected["month"], 1)
        #     return (
        #         0.4 + 0.1 * int(predicted["day"] is None)
        #     ) if delay.days < 60 else 0, TRESHOLD
        # return 0, TRESHOLD

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DataInterventoOutputData]:
        def to_date(date: str | None) -> DataInterventoOutputData:
            # TODO:
            if date is None:
                # TODO:
                raise NotImplementedError
            normalized_date = date.lower()

            # WARNING: the date string has not extact pattern
            # TODO: fuzzy match 0, 1 or 2 days and infer the precision and the
            # window

            # TODO: fuzzy match 0, 1 or 2 months and infer the precision and
            # the window
            def compute_months(val: str):
                found_months = [
                    idx + 1 for idx, m in enumerate(MONTHS) if m.lower() in val
                ]
                found_months = found_months
                raise NotImplementedError

            def compute_years(val: str):
                years = re.findall(r"\d{4}", val)
                if not years:
                    return -1, -1
                if len(years) == 1:
                    return years[0], years[0]
                if len(years) == 2:
                    return years[0], years[1]
                # impossible to know
                return -1, -1

            compute_months(normalized_date)
            compute_years(normalized_date)
            raise NotImplementedError

        return {
            InterventionId(t.id): to_date(t.university__Data_intervento)
            for t in y.get_answers(ids)
        }
