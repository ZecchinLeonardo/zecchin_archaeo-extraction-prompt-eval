"""Comune LLM extractor."""

from typing import Literal, Optional, TypedDict, cast, override

import dspy
import pandera.pandas as pa
import pydantic
from pandera.typing.pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.types.intervention_id import InterventionId

from ....types.per_intervention_feature import BasePerInterventionFeatureSchema
from ..field_extractor import FieldExtractor, TypedDspyModule

# -- DSPy part


# TODO: describe the model in Italian for the dspy model
class Comune(pydantic.BaseModel):
    """Questo elemento fornisce informazioni su un comune. È possibile trovare questo tipo di informazioni nel testo."""

    citta_nome: str
    provicia_nome: str
    provincia_sigla: str


MONTHS = [
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


class StimareDataDellIntervento(dspy.Signature):
    """Stima il momento di partenza dell'intervento con un'accurata data o un approssimato massimo meso di un anno."""

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    mese: Month = dspy.OutputField(
        desc='Rispondere "Dicembre" se non so il mese.'
    )
    anno: int = dspy.OutputField()
    giorno: Optional[int] = dspy.OutputField()  # noqa: UP045


class DataInterventoInputData(TypedDict):
    """Chunks of reports of an archaeological intervention with supposed information about the date of the intervention."""

    fragmenti_relazione: str


class DataInterventoOutputData(TypedDict):
    """A predicted maximum date for the intervention."""

    day: int | None
    month: int  # between 1 and 12
    year: int


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

    def forward(self, fragmenti_relazione: str):
        """Simple date parsing."""
        result = cast(
            dspy.Prediction,
            self._estrattore_della_data(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )
        got_mese = cast(str | None, result.get("mese"))
        if got_mese is None or got_mese not in MONTHS:
            got_mese = "Dicembre"
        return dspy.Prediction(
            **DataInterventoOutputData(
                day=result.get("giorno"),
                month=MONTHS.index(got_mese),
                year=cast(int, result.get("anno", 0)),
            )
        )


# -- SKlearn part


class DateFeatSchema(BasePerInterventionFeatureSchema):
    """Extracted data about the intervention start date."""

    intervention_date: Series[pa.DateTime]


class ComuneExtractor(
    FieldExtractor[
        DataInterventoInputData,
        DataInterventoOutputData,
        str,
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
            ),
            DataInterventoOutputData(day=18, month=3, year=1985),
        )
        super().__init__(
            llm_model,
            EstimateInterventionDate(),
            example,
        )

    @override
    def _to_dspy_input(self, x) -> DataInterventoInputData:
        return DataInterventoInputData(
            fragmenti_relazione=x.merged_chunks,
        )

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        if predicted == expected:
            return 1, TRESHOLD
        if predicted["month"] == expected["month"] and predicted["year"] == expected["year"]:
            if expected["day"] is None:
                return 0.9, TRESHOLD
            if predicted["day"] is None:
                return 0.7, TRESHOLD
            return 0.7 if predicted["day"] > expected["day"] else 0.6, TRESHOLD
        # TODO: if an approximated period under 3 months after the actual date
        # is predicted, then we give 0.5. Else, we return 0
        return 0, TRESHOLD

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DataInterventoOutputData]:
        def to_date_intervento(comune_string: str | None) -> ComuneOutputData:
            default_output = ComuneOutputData(comune="", provincia="")
            if comune_string is None:
                return default_output
            pattern = r"^(.*?) \((.*?)\)$"
            match = re.match(pattern, comune_string)
            if match:
                comune, provincia = match.groups()
                return ComuneOutputData(comune=comune, provincia=provincia)
            return default_output

        return {
            InterventionId(t.id): to_comune_data(t.university__Comune)
            for t in y.get_answers(ids)
        }
