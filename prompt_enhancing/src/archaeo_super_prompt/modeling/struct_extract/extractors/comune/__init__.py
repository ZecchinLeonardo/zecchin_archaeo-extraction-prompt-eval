"""Comune LLM extractor."""

import re
from typing import cast, override

import dspy
import pydantic
from pandera.typing.pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.dataset.thesauri import load_comune_with_provincie
from archaeo_super_prompt.modeling.struct_extract.types import (
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...field_extractor import FieldExtractor, LLMProvider, TypedDspyModule

# -- DSPy part


class Comune(pydantic.BaseModel):
    """Questo elemento fornisce informazioni su un comune. È possibile trovare questo tipo di informazioni nel testo."""

    citta_nome: str
    provicia_nome: str
    provincia_sigla: str


class IdentificaComune(dspy.Signature):
    """Identifica il unico comune in cui si sono svolti i lavori archeologici descritti in questi frammenti di relazione. I comuni possibili sono indicati."""

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    possibili_comuni: list[Comune] = dspy.InputField(
        desc="Scegliete un di questi comuni"
    )
    comune: str = dspy.OutputField(desc="Il nome completo del comune")
    provincia: str = dspy.OutputField(desc="Il nome completo della provincia")


class ComuneInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the comune where the operations took place.

    Identified likely comuni with their province are also provided to help in the extraction.
    """

    fragmenti_relazione: str
    possibili_comuni: list[Comune]


class ComuneOutputData(pydantic.BaseModel):
    """A predicted comune where the intervention took place, with its provincia."""

    comune: str
    provincia: str


class FindComune(TypedDspyModule[ComuneInputData, ComuneOutputData]):
    """DSPy model for the extraction of the comune."""

    def __init__(self):
        """Initialize only a chain of thought."""
        super().__init__(ComuneOutputData)
        self._estrattore_di_comune = dspy.ChainOfThought(IdentificaComune)

    def forward(
        self, fragmenti_relazione: str, possibili_comuni: list[Comune]
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_di_comune(
                fragmenti_relazione=fragmenti_relazione,
                possibili_comuni=possibili_comuni,
            ),
        )
        WRONG_COMUNE = "%ERROR_COMUNE%"
        WRONG_PROVINCIA = "%ERROR_PROVINCIA%"
        return self._to_prediction(
            ComuneOutputData(
                comune=cast(str, predicted_output.get("comune", WRONG_COMUNE)),
                provincia=cast(
                    str, predicted_output.get("provincia", WRONG_PROVINCIA)
                ),
            )
        )


# -- SKlearn part


class ComuneFeatSchema(BasePerInterventionFeatureSchema):
    """Extracted data about the Comune."""

    comune_id: int
    provincia_id: int


class ComuneExtractor(
    FieldExtractor[
        ComuneInputData,
        ComuneOutputData,
        InputForExtractionWithSuggestedThesauri,
        InputForExtractionWithSuggestedThesauriRowSchema,
        ComuneFeatSchema,
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
            ComuneInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
L'evento si è svolto a Lucca.""",
                possibili_comuni=[
                    Comune(
                        citta_nome="Lucca",
                        provicia_nome="Lucca",
                        provincia_sigla="LU",
                    )
                ],
            ),
            ComuneOutputData(comune="Lucca", provincia="Lucca"),
        )
        # TODO: load this more lazily
        self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindComune(),
            example,
            ComuneOutputData,
        )

    @override
    def _to_dspy_input(self, x) -> ComuneInputData:
        comuni, province = self._thesaurus
        possible_comuni = comuni.iloc[x.identified_thesaurus].merge(
            province, on="province_id", suffixes=("_comune", "_province")
        )
        return ComuneInputData(
            fragmenti_relazione=x.merged_chunks,
            possibili_comuni=[
                Comune(
                    citta_nome=cast(str, c.name_comune),
                    provicia_nome=cast(str, c.name_province),
                    provincia_sigla=cast(str, c.sigla),
                )
                for c in possible_comuni.itertuples()
            ],
        )

    @override
    def _transform_dspy_output(self, y):
        comuni, province = self._thesaurus
        return ComuneFeatSchema.validate(
            self._identity_output_set_transform_to_df(y)
            .assign(schedaid=lambda df: df.index)
            .merge(
                province[["name"]].assign(provincia_id=province.index),
                left_on="provincia",
                right_on="name",
            )[["schedaid", "comune", "provincia_id"]]
            .merge(
                comuni.assign(comune_id=comuni.index),
                left_on=["comune", "provincia_id"],
                right_on=["name", "province_id"],
            )[["schedaid", "comune_id", "provincia_id"]]
            .rename(columns={"schedaid": "id"})
            .set_index("id"),
            # TODO: add this after tests
            # lazy=True
        )

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        TRESHOLD = 0.95
        return 0.7 * int(predicted.comune == expected.comune) + 0.3 * int(
            predicted.provincia == expected.provincia
        ), TRESHOLD

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Comune"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, ComuneOutputData]:
        def to_comune_data(comune_string: str | None) -> ComuneOutputData:
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

    @override
    @staticmethod
    def field_to_be_extracted():
        return "comune"
