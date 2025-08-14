"""LLM-based extraction of the date of start of the intervention.

We expect the model to predict a window of dates for the start of the
intervention. This model has a known/guessed date of archiving of the report
and can output a window at least before this date.

The precision of the window is among those below :
1. Day
2. Month
3. Year
Moreover, the earlier date in the window can be open if the information is not
guessable. The most recent in the window must be by default the date of
archiving if the information is unknown.
"""

import datetime
from typing import Literal, Optional, cast, override

import dspy
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series
import pydantic

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.types import (
    BaseInputForExtraction,
    BaseInputForExtractionRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...field_extractor import FieldExtractor, LLMProvider, to_prediction
from .type_models import ITALIAN_MONTHS, Data, Precision, Precisione


# -- DSPy part


class StimareDataDellIntervento(dspy.Signature):
    """Degli framenti datti di relazione archeologiche, stima il momento di partenza dell'indagine in una finestra de due date, con un precisione al giorno, al mese o all'anno più vicino. Se non ci sono molte informazione, ritorna almeno una finestra prima di la data di archiviazone datta. La data minima può essere lasciata sconosciuta.

    1. Innanzitutto, determina se c'è una data minima e la precisione con cui puoi approssimare la finestra.
    2. Quindi, determina la finestra, inserendo valori predefiniti (ma ben tipizzati) nei campi non coperti dalla precisione.
       a. Se possibile, restringi la finestra a un punto impostando le stesse date minima e massima.
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    data_di_archiviazone: Data = dspy.InputField()

    data_minima_di_inizio: Optional[Data] = dspy.OutputField()  # noqa: UP045
    data_massima_di_inizio: Data = dspy.OutputField()
    precisione: Precisione = dspy.OutputField()


class DataInterventoInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the date of the intervention."""

    fragmenti_relazione: str
    data_di_archiviazone: Data


class DataInterventoOutputData(pydantic.BaseModel):
    """A predicted maximum date for the intervention, in un window."""

    start_day: int | None
    start_month: int | None  # between 1 and 12
    start_year: int | None
    end_day: int
    end_month: int  # between 1 and 12
    end_year: int
    precision: Precision


def _get_min_date(output_model: DataInterventoOutputData):
    return (
        datetime.date(
            output_model.start_year,
            output_model.start_month,
            output_model.start_day,
        )
        if (
            output_model.start_year is not None
            and output_model.start_month is not None
            and output_model.start_day is not None
        )
        else None
    )


def _get_max_date(output_model: DataInterventoOutputData):
    return datetime.date(
        output_model.end_year, output_model.end_month, output_model.end_day
    )


class EstimateInterventionDate(
    dspy.Module
):
    """DSPy model for the extraction of the date of the intervention."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_della_data = dspy.ChainOfThought(
            StimareDataDellIntervento
        )

    def forward(
        self, fragmenti_relazione: str, data_di_archiviazone: Data
    ) -> dspy.Prediction:
        """Simple date parsing."""
        result = cast(
            dspy.Prediction,
            self._estrattore_della_data(
                fragmenti_relazione=fragmenti_relazione,
                data_di_archiviazone=data_di_archiviazone,
            ),
        )

        DEFAULT_WRONG_DATE = Data(
            giorno=25, mese="Dicembre", anno=1
        )  # The child was born
        TO_ENGLISH_PRECISION: dict[Precisione, Precision] = {
            "giorno": "day",
            "mese": "month",
            "anno": "year",
        }

        data_minima_di_inizio = cast(
            Data | None, result.get("data_minima_di_inizio")
        )
        data_massima_di_inizio = cast(
            Data, result.get("data_massima_di_inizio", DEFAULT_WRONG_DATE)
        )
        precisione = cast(
            Literal["giorno", "mese", "anno"],
            result.get("precisione", "giorno"),
        )

        return to_prediction(
            DataInterventoOutputData(
                start_day=data_minima_di_inizio.giorno
                if data_minima_di_inizio is not None
                else None,
                start_month=ITALIAN_MONTHS.index(data_minima_di_inizio.mese)
                if data_minima_di_inizio is not None
                else None,
                start_year=data_minima_di_inizio.anno
                if data_minima_di_inizio is not None
                else None,
                end_day=data_massima_di_inizio.giorno,
                end_month=ITALIAN_MONTHS.index(data_massima_di_inizio.mese),
                end_year=data_massima_di_inizio.anno,
                precision=TO_ENGLISH_PRECISION[precisione],
            )
        )


# -- SKlearn part


class InputForInterventionDate(BaseInputForExtraction):
    """When indentifying the date of an intervention, we refer first to the date of protocol."""

    data_protocollo: datetime.date


class InputForInterventionDateRowSchema(BaseInputForExtractionRowSchema):
    """When indentifying the date of an intervention, we refer first to the date of protocol."""

    data_protocollo: datetime.date


class DateFeatSchema(BasePerInterventionFeatureSchema):
    """Extracted data about the intervention start date."""

    intervention_start_date_min: Optional[datetime.date] = pa.Field(  # noqa: UP045
        nullable=True
    )
    intervention_start_date_max: datetime.date
    intervention_start_date_precision: Series[str] = (
        pa.Field(isin=["day", "month", "year"])
    )


class InterventionStartExtractor(
    FieldExtractor[
        DataInterventoInputData,
        DataInterventoOutputData,
        InputForInterventionDate,
        InputForInterventionDateRowSchema,
        DateFeatSchema,
    ]
):
    """Dspy-LLM-based extractor of the comune data."""

    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
    ) -> None:
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
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            EstimateInterventionDate(),
            example,
            DataInterventoOutputData,
        )

    @override
    def _to_dspy_input(self, x) -> DataInterventoInputData:
        date_of_archiving = x.data_protocollo
        return DataInterventoInputData(
            fragmenti_relazione=x.merged_chunks,
            data_di_archiviazone=Data(
                giorno=date_of_archiving.day,
                mese=ITALIAN_MONTHS[date_of_archiving.month],
                anno=date_of_archiving.year,
            ),
        )

    @override
    def _transform_dspy_output(self, y):
        return DateFeatSchema.validate(
            pd.DataFrame(
                [
                    {
                        "id": id_,
                        "intervention_start_date_min": _get_min_date(y),
                        "intervention_start_date_max": _get_max_date(y),
                        "intervention_start_date_precision": y.precision,
                    }
                    for id_, y in y
                ]
            ).set_index("id")
            # TODO: add this argument
            # lazy=True,
        )

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        def compare_dates(
            predicted: datetime.date,
            expected: datetime.date,
            expected_precision: Precision,
        ) -> float:
            if predicted.year != expected.year:
                return 0.0
            if expected_precision == "year":
                return 1.0
            if predicted.month != expected.month:
                return 0.5
            if expected_precision == "month":
                return 1.0
            return 1.0 if predicted.day == expected.day else 0.6

        precision_reward_coeff = (
            1.0 if predicted.precision == expected.precision else 0.6
        )
        expected_max = _get_max_date(expected)
        predicted_max = _get_max_date(predicted)
        expected_min = _get_min_date(expected)
        predicted_min = _get_min_date(expected)
        ref_precision = expected.precision

        score_for_min_date = 0.0
        if expected_min is None or predicted_min is None:
            score_for_min_date = 1.0 if expected_min == predicted_min else 0.0
        else:
            score_for_min_date = compare_dates(
                predicted_min, expected_min, ref_precision
            )

        score_for_max_date = compare_dates(
            predicted_max, expected_max, ref_precision
        )
        TRESHOLD = 0.9

        return (
            score_for_min_date + score_for_max_date
        ) * precision_reward_coeff / 2, TRESHOLD

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DataInterventoOutputData]:
        def to_data(
            start: datetime.date | None, end: datetime.date, precision: str
        ) -> DataInterventoOutputData:
            sd, sm, sy = None, None, None
            if start is not None:
                sd, sm, sy = start.day, start.month, start.year
            return DataInterventoOutputData(
                start_day=sd,
                start_month=sm,
                start_year=sy,
                end_day=end.day,
                end_month=end.month,
                end_year=end.year,
                precision=cast(Precision, precision),
            )

        return {
            InterventionId(answer.id): to_data(
                answer.intervention_start_date_min,
                answer.intervention_start_date_max,
                answer.intervention_start_date_precision,
            )
            for answer in y.get_answers(ids)
        }

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        # we filter nothing as the correct start intervention date has always
        # be figured out by the Magoh Contributors on the featured dataset
        y = y  # unused
        return ids

    @override
    @staticmethod
    def field_to_be_extracted():
        return "intervention-start-date"
