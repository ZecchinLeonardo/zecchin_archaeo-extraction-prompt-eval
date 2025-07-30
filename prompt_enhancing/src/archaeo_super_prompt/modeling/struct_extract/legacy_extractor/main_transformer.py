"""SKLearn Transformer for the first polyvalent extraction model."""

from typing import cast, override

from dspy import Example, Prediction, dspy

import pandas
from pandera.typing.pandas import DataFrame

from archaeo_super_prompt.modeling.types.detailed_evaluator import (
    DetailedEvaluatorMixin,
)

from .evaluation.compare import validate_magoh_data
from ....types.results import ResultSchema

from ....dataset.load import MagohDataset
from ....config.debug_log import print_log
from .evaluation.evaluate import get_evaluator
from .evaluation.load_examples import DevSet
from .extractor_module import (
    ExtractDataFromInterventionReport,
)
from ....types.intervention_id import InterventionId


from ....types.pdfchunks import (
    PDFChunkDataset,
    PDFChunkPerInterventionDataset,
    PDFChunkSetPerInterventionSchema,
)
from ....types.structured_data import OutputStructuredDataSchema


class MagohDataExtractor(
    DetailedEvaluatorMixin[
        PDFChunkDataset, MagohDataset, DataFrame[ResultSchema]
    ]
):
    """Main model extracting structured data from contextualized LLM prompts.

    It is a dspy model that can be trained and scored.
    """

    def __init__(self, llm: dspy.LM) -> None:
        """The main hyperparametre is the temperature of the llm model."""
        super().__init__()
        self._module = ExtractDataFromInterventionReport()
        self._llm = llm

    @property
    def dspy_model(self):
        """The dspy model, for logging in mlflow."""
        return self._module

    def compute_model_input(self, X: PDFChunkDataset):
        """Transform the dataframe into an Iterable of input for the dspy module."""
        return [
            (
                id_,
                PDFChunkPerInterventionDataset(
                    PDFChunkSetPerInterventionSchema.validate(
                        source, lazy=True
                    ),
                ),
            )
            for id_, source in X.groupby("id")
        ]

    def compute_devset(
        self, X: PDFChunkDataset, targets: MagohDataset
    ) -> DevSet:
        """Compute a set of dspy example for an evaluation or an optimization."""
        return [
            dspy.Example(
                document_ocr_scans__df=model_input,
                **targets.get_answer(InterventionId(cast(int, id_))),
            ).with_inputs("document_ocr_scans__df")
            for id_, model_input in self.compute_model_input(X)
        ]

    # TODO: code an optimization in overriding the fit method

    @override
    def transform(
        self, X: PDFChunkDataset
    ) -> DataFrame[OutputStructuredDataSchema]:
        with dspy.settings.context(lm=self._llm):
            answers = {
                id_: answer
                for id_, answer in {
                    id_: self._module.forward_and_type(model_input)
                    for id_, model_input in self.compute_model_input(X)
                }.items()
                if answer is not None
            }
            ids, output_structured_data = zip(*answers.items())
            answer_df = pandas.DataFrame(
                cast(list[dict], list(output_structured_data))
            )
            answer_df["id"] = cast(list[InterventionId], list(ids))
            return OutputStructuredDataSchema.validate(
                answer_df.astype(
                    {
                        "university__Numero_di_saggi": "UInt32",
                        "university__Geologico": "boolean",
                        "university__Profondità_falda": "Float64",
                        "university__Profondità_massima": "Float64",
                    }
                )
            )

    @override
    def score_and_transform(self, X: PDFChunkDataset, y: MagohDataset):
        """Run an evaluation of the dpsy model over the given X dataset.

        Return the per-field results for each test record in a dataframe.

        To fit the sklearn Classifier interface, this method return a reduced
        floating metric value for the model.
        """
        devset = self.compute_devset(X, y)

        # The evaluator only enable to automate the standard workflow of dspy
        # for running evaluation inferences but this workflow is not suitable
        # for a per-field evaluation
        evaluate = get_evaluator(devset, return_outputs=True)
        print_log("Tracing ready!\n")
        with dspy.settings.context(lm=self._llm):
            results = cast(
                tuple[float, list[tuple[Example, Prediction, float]]],
                evaluate(self._module),
            )
            return (
                results[0],
                ResultSchema.validate(
                    pandas.DataFrame(
                        sum(
                            [
                                [
                                    {
                                        "id": ex.get("id"),
                                        "field_name": field,
                                        "predicted_value": pred.get(field),
                                        "expected_value": ex.get(field),
                                        "evaluation_method": "not specified yet",  # TODO:
                                        "metric_value": float(metric_value),
                                    }
                                    for field, metric_value in validate_magoh_data(
                                        ex, pred
                                    ).items()
                                ]
                                for ex, pred, _ in results[1]
                            ],
                            [],
                        )
                    ),
                ),
            )

    @override
    def score(self, X, y, sample_weight=None):
        sample_weight = sample_weight  # unused
        score, _ = self.score_and_transform(X, y)
        return score
