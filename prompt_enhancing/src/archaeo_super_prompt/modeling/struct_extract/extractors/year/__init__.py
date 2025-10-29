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
import re

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.types import (
    BaseInputForExtraction,
    BaseInputForExtractionRowSchema,
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...field_extractor import FieldExtractor, LLMProvider, to_prediction
from .type_models import ITALIAN_MONTHS, Data, Precision, Precisione


# -- DSPy part


class StimareData(dspy.Signature):
    """Dai framenti di relazione archeologiche, stima l'anno in cui si è svolta l'indagine archeologica.

    Cerca una stringa come "lo scavo è iniziato il", "la ricognizione è iniziata il", "l'indagine è iniziata il", "lo scavo è terminato il", "la ricognizione è terminata il", "l'indagine è terminata il".

    Se c'è una data questa è solitamente in formato "18 marzo 1985", "18/03/1985", "18-03-1985", "marzo 1985", "03/1985", "03-1985", "1985".

    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    data_intervento: Data = dspy.OutputField()


class DataInterventoInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the date of the intervention."""

    fragmenti_relazione: str


class DataInterventoOutputData(pydantic.BaseModel):
    """A predicted year."""

    data: Data  # Use the Data type from type_models.py
    year: int | None

class EstimateData(
    dspy.Module
):
    """DSPy model for the extraction of the year from the year of the intervention."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_data = dspy.ChainOfThought(
            StimareData
        )

    def forward(
        self, fragmenti_relazione: str
    ) -> dspy.Prediction:
        """Simple date parsing."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_data(
                fragmenti_relazione=fragmenti_relazione,
            ),
        )

        DATA_UNIDENTIFIED = Data(giorno=1, mese="Gennaio", anno=0)
        pred_data = cast(Data, predicted_output.get("data_intervento", DATA_UNIDENTIFIED))
        # print(f"Predicted data: {pred_data}, type: {type(pred_data)}")

        return to_prediction(
            DataInterventoOutputData(
                data=pred_data,
                year = pred_data.anno,
            )
        )


# -- SKlearn part
class InputForInterventionDate(BaseInputForExtraction):
    """When indentifying the date of an intervention, we refer first to the date of protocol."""

    data_intervento: datetime.date


class InputForInterventionDateRowSchema(BaseInputForExtractionRowSchema):
    """When indentifying the date of an intervention, we refer first to the date of protocol."""

    data_intervento: datetime.date

class YearExtractor(
    FieldExtractor[
        DataInterventoInputData,
        DataInterventoOutputData,
        InputForInterventionDate,
        InputForInterventionDateRowSchema,

        None,

        
        # DateFeatSchema,
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
            ),
            DataInterventoOutputData(
                data=Data(giorno=18, mese="Marzo", anno=1985),
                year=1985,
            ),
        )
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            EstimateData(),
            example,
            DataInterventoOutputData,
        )

    @override
    def _to_dspy_input(self, x) -> DataInterventoInputData:
        return DataInterventoInputData(
            fragmenti_relazione=x.merged_chunks,
        )

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected ProtocolloOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        data_pred = dspy_output.get("data_intervento") or dspy_output.get("pred_data_intervento") or None
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want


        return DataInterventoOutputData(
            data=data_pred,
            year = data_pred.anno if data_pred else None,
            method=method  # Only include this if your schema expects it!
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__Data intervento"
        
    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95

        score = float(str(predicted.year) == str(expected.year))

        return score, TRESHOLD

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Data_intervento"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DataInterventoOutputData]:
 
        result = {}
        for t in y.get_answers(ids):
            if t.university__Data_intervento is not None:  # Skip if no ground truth
                data_intervento = t.university__Data_intervento
                print(f"Raw data_intervento: {data_intervento}, type: {type(data_intervento)}")
                # Convert to Data class type from type_models
                if isinstance(data_intervento, Data):
                    data_obj = data_intervento
                elif isinstance(data_intervento, str):
                    data_obj = cls.parse_data_intervento(data_intervento)
                # exp_year = cls.extract_year(str(data_intervento))

                result[InterventionId(t.id)] = DataInterventoOutputData(
                    data=data_obj,
                    year=data_obj.anno if data_obj else None,
                )
    
        return result
    
    @staticmethod
    def parse_data_intervento(data_intervento):
        ITALIAN_MONTHS = [
            "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
            "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
        ]
        if isinstance(data_intervento, Data):
            return data_intervento
        elif isinstance(data_intervento, str):
            stripped = data_intervento.strip()

            # Special case: handle "pre, 1987", "pre -, 1980", etc.
            pre_match = re.match(r"(?:pre|ante)[\s,-]*([\d]{4})", stripped, re.IGNORECASE)
            if pre_match:
                anno = int(pre_match.group(1))
                return Data(giorno=1, mese="Gennaio", anno=anno)

            # General case: find all date-like patterns (start date of range)
            date_pattern = re.compile(
                r"(?:(\d{1,2})(?:-\d{1,2})?\s*)?"        # day or day-interval (optional)
                r"([A-Za-zàèéìòù]+|\d{1,2})?"            # month as word or number (optional)
                r"[\s,/-]*"
                r"(\d{4})",                              # year (required)
                re.IGNORECASE,
            )
            all_matches = list(date_pattern.finditer(stripped))
            if all_matches:
                match = all_matches[0]
                groups = match.groups()
                giorno = None
                mese = None
                anno = None
                if groups[2]:
                    try:
                        anno = int(groups[2])
                    except Exception:
                        anno = 0
                if groups[1]:
                    try:
                        mese_int = int(groups[1])
                        if 1 <= mese_int <= 12:
                            mese = ITALIAN_MONTHS[mese_int - 1]
                    except Exception:
                        # m = groups[1].capitalize()
                        # for m_it in ITALIAN_MONTHS:
                        #     if m_it.lower() == m.lower():
                        #         mese = m_it
                        #         break
                        # else:
                        #     mese = m
                        # textual month: normalize and try to match known italian months
                            m = groups[1].strip().capitalize()
                            # remove trailing punctuation
                            m = re.sub(r"[^\wàèéìòù]", "", m, flags=re.UNICODE)
                            for m_it in ITALIAN_MONTHS:
                                if m_it.lower() == m.lower():
                                    mese = m_it
                                    break
                            # try partial match (e.g. "giu" -> "Giugno")
                            if mese is None:
                                m_lower = m.lower()
                                for m_it in ITALIAN_MONTHS:
                                    if m_it.lower().startswith(m_lower) or m_lower.startswith(m_it.lower()[:3]):
                                        mese = m_it
                                        break
                            # if still unknown, leave mese None (will fallback later)

                if groups[0]:
                    try:
                        giorno = int(groups[0])
                    except Exception:
                        giorno = None
                if not mese:
                    mese = "Gennaio"
                if not giorno:
                    giorno = 1
                if not anno:
                    anno = 0
                return Data(giorno=giorno, mese=mese, anno=anno)
            else:
                # fallback for raw year only
                year_match = re.search(r"\d{4}", stripped)
                if year_match:
                    return Data(giorno=1, mese="Gennaio", anno=int(year_match.group(0)))
                else:
                    return Data(giorno=1, mese="Gennaio", anno=0)
        elif isinstance(data_intervento, datetime.date):
            return Data(
                giorno=data_intervento.day,
                mese=ITALIAN_MONTHS[data_intervento.month - 1],
                anno=data_intervento.year,
            )
        else:
            return Data(giorno=1, mese="Gennaio", anno=0)
