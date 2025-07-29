"""Comune LLM extractor."""

from typing import override
import dspy
import re
import pydantic

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.dataset.thesaurus import load_comune_with_provincie
from archaeo_super_prompt.modeling.struct_extract.types import (
    InputForExtractionWithSuggestedThesauri,
    InputForExtractionWithSuggestedThesauriRowSchema,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

from .....types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)

from ...field_extractor import FieldExtractor, TypedDspyModule

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
    ):
        """Direct forward."""
        # the signature's output is the same output; so we directly return
        return self._estrattore_di_comune(
            fragmenti_relazione=fragmenti_relazione,
            possibili_comuni=possibili_comuni,
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

    def __init__(self, llm_model: dspy.LM) -> None:
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
            llm_model,
            FindComune(),
            example,
            ComuneOutputData,
        )

    @override
    def _to_dspy_input(self, x) -> ComuneInputData:
        possible_comuni = [
            self._thesaurus[th_id] for th_id in x.identified_thesaurus
        ]
        return ComuneInputData(
            fragmenti_relazione=x.merged_chunks,
            possibili_comuni=[
                Comune(
                    citta_nome=c.comune,
                    provicia_nome=c.provincia.name,
                    provincia_sigla=c.provincia.sigla,
                )
                for c in possible_comuni
            ],
        )

    @override
    def _transform_dspy_output(self, y):
        # DEBUG
        print("Here we are")
        return ComuneFeatSchema.validate(
            self._identity_output_set_transform_to_df(y),
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
            ids, lambda df: df["university__Comune"].notnull()
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
