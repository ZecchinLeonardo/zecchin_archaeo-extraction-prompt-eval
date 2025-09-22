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
from ...field_extractor import FieldExtractor, LLMProvider, to_prediction

# -- DSPy part


class Esecuzione(pydantic.BaseModel):
    """Questo elemento fornisce informazioni sulla persona che ha eseguito il lavoro. È possibile trovare questo tipo di informazioni nel testo."""

    esecutore: str
    


class IdentificaEsecutore(dspy.Signature):
    """Identifica la persona che ha eseguito i lavori archeologici descritti in questi frammenti di relazione.
    
    Cerca una stringa come "Eseguito da"
    """

    fragmenti_relazione: str = dspy.InputField(
        desc="In ogni frammento sono indicati il nome del file pdf e la sua posizione nel file."
    )

    # possibili_esecutori: list[Esecuzione] = dspy.InputField(
    #     desc="Scegliete una di queste persone che hanno eseguito i lavori archeologici."
    # )

    esecutore: str = dspy.OutputField(desc="Il nome completo di chi ha eseguito i lavori.")


class EsecutoreInputData(pydantic.BaseModel):
    """Chunks of reports of an archaeological intervention with supposed information about the person who carried out the operations.
    
    Find in the text a string like "Eseguito da"
    """

    fragmenti_relazione: str
    # possibili_esecutori: list[Esecuzione]


class EsecutoreOutputData(pydantic.BaseModel):
    """A predicted person who performed the intervention."""

    nome_cognome: str
    #cognome: str


class FindEsecutore(dspy.Module):
    """DSPy model for the extraction of the Esecutore."""

    def __init__(self):
        """Initialize only a chain of thought."""
        self._estrattore_esecuzione = dspy.ChainOfThought(IdentificaEsecutore)

    def forward(
        self, fragmenti_relazione: str#, possibili_esecutori: list[Esecuzione]
    ) -> dspy.Prediction:
        """Direct forward."""
        predicted_output = cast(
            dspy.Prediction,
            self._estrattore_esecuzione(
                fragmenti_relazione=fragmenti_relazione,
                #possibili_esecutori=possibili_esecutori,
            ),
        )
        # WRONG_COMUNE = "%ERROR_COMUNE%"
        # WRONG_PROVINCIA = "%ERROR_PROVINCIA%"
        UNIDENTIFIED = "%CHECK REQUIRED%"
        return to_prediction(
            EsecutoreOutputData(
                nome_cognome=cast(str, predicted_output.get("esecutore", UNIDENTIFIED)),
                # provincia=cast(
                #     str, predicted_output.get("provincia", WRONG_PROVINCIA)
                # ),
            )
        )


# -- SKlearn part


# class ComuneFeatSchema(BasePerInterventionFeatureSchema):
#     """Extracted data about the Comune."""

#     comune_id: int
#     provincia_id: int


class EsecutoreExtractor(
    FieldExtractor[
        EsecutoreInputData,
        EsecutoreOutputData,
        InputForExtractionWithSuggestedThesauri,
        InputForExtractionWithSuggestedThesauriRowSchema,
        None,
        # ComuneFeatSchema,
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
            EsecutoreInputData(
                fragmenti_relazione=""""Relazione_scavo.pdf, Pagina 1 :
L'intervento è stato eseguito da Mario Rossi in data 12/05/2023.""",
                # possibili_esecutori=[
                #     Esecuzione(
                #         esecutore="Mario Rossi",
                #         # provicia_nome="Lucca",
                #         # provincia_sigla="LU",
                #     )
                # ],
            ),
            EsecutoreOutputData(nome_cognome="Mario Rossi"),
        )
        # TODO: load this more lazily
        # self._thesaurus = load_comune_with_provincie()
        super().__init__(
            llm_model_provider,
            llm_model_id,
            llm_temperature,
            FindEsecutore(),
            example,
            EsecutoreOutputData,
        )

    @override
    @staticmethod
    def field_to_be_extracted():
        # Return the exact field name you want to extract
        return "university__Eseguito_da"

    # def _compare_values(self, a, b):
    #     # Implement logic to compare two values for this field (ground truth vs prediction)
    #     # Simple example:
    #     return a == b
        
    @override
    @classmethod
    def _compare_values(cls, a, b):
        score = int(a == b)  # 1 if match, 0 otherwise
        return score, 0.95


    # def _transform_dspy_output(self, dspy_output):
    #     # Transform the raw output from your prompt/model into your output schema
    #     # Example (adapt as needed to your EsecutoreOutputData):
    #     return EsecutoreOutputData(**dspy_output)
    
    # @override
    # def _transform_dspy_output(self, dspy_output):
    #     # Defensive: get 'esecutore', fallback to empty string
    #     return EsecutoreOutputData(
    #         nome_cognome=dspy_output.get("esecutore", "")
    #     )

    @override
    def _transform_dspy_output(self, dspy_output):
        """
        Remap the DSPy output to the expected EsecutoreOutputData schema.
        If the output is missing or has unexpected keys, handle gracefully.
        """
        # Defensive mapping: look for common keys, fallback to empty string or 'N/A'
        nome = dspy_output.get("esecutore") or dspy_output.get("nome_cognome") or dspy_output.get("pred_nome_cognome") or ""
        method = dspy_output.get("method", "LLM")  # You can set this to whatever method name you want

        return EsecutoreOutputData(
            nome_cognome=nome,
            method=method  # Only include this if your schema expects it!
        )

    
    # @override
    # def _to_dspy_input(self, x) -> EsecutoreInputData:
    #     # comuni, province = self._thesaurus
    #     # possible_comuni = comuni.iloc[x.identified_thesaurus].merge(
    #     #     province, on="province_id", suffixes=("_comune", "_province")
    #     # )

    #     return EsecutoreInputData(
    #         fragmenti_relazione=x.merged_chunks,
    #         # possibili_comuni=[
    #         #     Comune(
    #         #         citta_nome=cast(str, c.name_comune),
    #         #         provicia_nome=cast(str, c.name_province),
    #         #         provincia_sigla=cast(str, c.sigla),
    #         #     )
    #         #     for c in possible_comuni.itertuples()
    #         # ],
    #     )

    @override
    def _to_dspy_input(self, x) -> EsecutoreInputData:
        # x is just a dict with 'id' (and maybe 'filepath')
        # You need to fetch the full row from the dataset
        intervention_id = x['id']
        full_row = self.dataset.intervention_data[self.dataset.intervention_data['id'] == intervention_id]
        if full_row.empty:
            # handle missing case
            return EsecutoreInputData(fragmenti_relazione="", esecutore_raw=None)
        row = full_row.iloc[0]
        return EsecutoreInputData(
            fragmenti_relazione=getattr(row, "merged_chunks", ""),
            esecutore_raw=getattr(row, "university__Eseguito_da", None),
        )

    # @override
    # def _transform_dspy_output(self, y):
    #     # comuni, province = self._thesaurus
    #     return ComuneFeatSchema.validate(
    #         self._identity_output_set_transform_to_df(y)
    #         .assign(schedaid=lambda df: df.index)
    #         .merge(
    #             province[["name"]].assign(provincia_id=province.index),
    #             left_on="provincia",
    #             right_on="name",
    #         )[["schedaid", "comune", "provincia_id"]]
    #         .merge(
    #             comuni.assign(comune_id=comuni.index),
    #             left_on=["comune", "provincia_id"],
    #             right_on=["name", "province_id"],
    #         )[["schedaid", "comune_id", "provincia_id"]]
    #         .rename(columns={"schedaid": "id"})
    #         .set_index("id"),
    #         # TODO: add this after tests
    #         # lazy=True
    #     )

    # @override
    # @classmethod
    # def _compare_values(cls, predicted, expected):
    #     TRESHOLD = 0.95
    #     return 0.7 * int(predicted.comune == expected.comune) + 0.3 * int(
    #         predicted.provincia == expected.provincia
    #     ), TRESHOLD

##############################################################################

    @override
    @classmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        return y.filter_good_records_for_training(
            ids,
            lambda df: cast(Series[bool], df["university__Eseguito_da"].notnull()),
        )

    @override
    @classmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, EsecutoreOutputData]:
        return {
            InterventionId(t.id): EsecutoreOutputData(nome_cognome=t.university__Eseguito_da)
            for t in y.get_answers(ids)
            if t.university__Eseguito_da is not None  # skip if no ground truth
        }

##############################################################################

    # @override
    # @staticmethod
    # def field_to_be_extracted():
    #     return "comune"
