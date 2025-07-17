"""Comune LLM extractor."""

from math import exp
from typing import TypedDict, override
import dspy
import pydantic

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.dataset.thesaurus import load_comune_with_provincie
from archaeo_super_prompt.types.intervention_id import InterventionId

from ....types.per_intervention_feature import BasePerInterventionFeatureSchema

from ..field_extractor import FieldExtractor, TypedDspyModule

# -- DSPy part


# TODO: describe the model in Italian for the dspy model
class Comune(pydantic.BaseModel):
    citta_nome: str
    provicia_nome: str
    provincia_sigla: str


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
        int,
        ComuneFeatSchema,
    ]
):
    def __init__(self, llm_model: dspy.LM) -> None:
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
        )

    @override
    def _to_dspy_input(self, x) -> ComuneInputData:
        possible_comuni = [
            self._thesaurus[th_id] for th_id in x.suggested_extraction_outputs
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
    @classmethod
    def _compare_values(cls, predicted, expected):
        return 0.7 * int(
            predicted["comune"] == expected["comune"]
        ) + 0.3 * int(predicted["provincia"] == expected["provincia"])

    @override
    @classmethod
    def select_answer(cls, y: MagohDataset, id: InterventionId) -> ComuneOutputData:
        # TODO:
        pass
