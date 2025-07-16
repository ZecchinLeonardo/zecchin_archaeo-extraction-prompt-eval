"""Comune LLM extractor."""

from typing import TypedDict, override
import dspy
import pydantic

from ....modeling.entity_extractor.types import ChunksWithThesaurus
from ....types.per_intervention_feature import BasePerInterventionFeatureSchema

from ..field_extractor import FieldExtractor, TypedDspyModule

# -- DSPy part


class Comune(pydantic.BaseModel):
    citta_nome: str
    provicia_nome: str
    provincia_siglo: str


class IdentificaComune(dspy.Signature):
    """Identifica il comune in cui si sono svolti i lavori archeologici descritti in questi frammenti di relazione. I comuni possibili sono indicati."""

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )
    possibili_comuni: list[Comune] = dspy.InputField(
        desc="Scegliete un di questi comuni"
    )
    comune: str = dspy.OutputField(desc="Il nome completo del comune")
    provincia: str = dspy.OutputField(desc="Il nome completo della provincia")


class ComuneInputData(TypedDict):
    fragmenti_relazione: str
    possibili_comuni: list[Comune]


class ComuneOutputData(TypedDict):
    comune: str
    provincia: str


class FindComune(TypedDspyModule[ComuneInputData, ComuneOutputData]):
    """DSPy model for the extraction of the comune."""

    def __init__(self, callbacks=None):
        """Initialize only a chain of thought."""
        super().__init__(callbacks)
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
        ChunksWithThesaurus,
        ComuneFeatSchema,
    ]
):
    def __init__(self) -> None:
        example = (
            ComuneInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
L'evento si è svolto a Lucca.""",
                possibili_comuni=[
                    Comune(
                        citta_nome="Lucca",
                        provicia_nome="Lucca",
                        provincia_siglo="LU",
                    )
                ],
            ),
            ComuneOutputData(comune="Lucca", provincia="Lucca"),
        )
        super().__init__(
            FindComune(),
            example,
        )

    @override
    @classmethod
    def _to_dspy_input(cls, X):
        return super()._to_dspy_input(X)

    @override
    @classmethod
    def _compare_values(cls, predicted, expected):
        return super()._compare_values(predicted, expected)

    @override
    def transform(self, X):
        return super().transform(X)
