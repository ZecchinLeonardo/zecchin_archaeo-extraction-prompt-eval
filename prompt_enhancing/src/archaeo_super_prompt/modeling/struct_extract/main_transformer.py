from typing import List, Optional, Tuple, cast

from dspy import Example, Prediction, dspy

import pandas
from pandera.typing.pandas import DataFrame

from .evaluation.compare import validate_magoh_data
from ...types.results import ResultSchema

from ...dataset.load import MagohDataset
from ...config.debug_log import print_log
from .evaluation.evaluate import get_evaluator
from .evaluation.load_examples import DevSet
from .extractor_module import (
    ExtractDataFromInterventionReport,
)
from ...types.intervention_id import InterventionId


from .language_model import load_model
from ...types.pdfchunks import (
    PDFChunkDataset,
    PDFChunkPerInterventionDataset,
    PDFChunkSetPerInterventionSchema,
)
from ...types.structured_data import outputStructuredDataSchema


class MagohDataExtractor:
    def __init__(self, llm_temp=0.0) -> None:
        self._module = ExtractDataFromInterventionReport()
        self._llm = load_model(llm_temp)
        self._module.set_lm(self._llm)
        self._cached_score_results: Optional[DataFrame[ResultSchema]] = None

    @property
    def score_results(self):
        return self._cached_score_results

    @property
    def dspy_model(self):
        return self._module

    def compute_model_input(self, X: PDFChunkDataset):
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
        return [
            dspy.Example(
                document_ocr_scans__df=model_input,
                **targets.get_answer(InterventionId(cast(int, id_))),
            ).with_inputs("document_ocr_scans__df")
            for id_, model_input in self.compute_model_input(X)
        ]

    def fit(self, X: PDFChunkDataset, Y: MagohDataset):
        # TODO: call a dspy optimizer
        X = X
        Y = Y
        return self

    # TODO: create a schema for the answer
    def transform(self, X: PDFChunkDataset) -> pandas.DataFrame:
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
        answer_df["id"] = cast(List[InterventionId], list(ids))
        return outputStructuredDataSchema.validate(answer_df)

    def score(self, X: PDFChunkDataset, targets: MagohDataset):
        """Run an evaluation of the dpsy model over the given X dataset

        Also save the per-field results for each test record in a cached
        dataframe, accessible after the function call with the score_results
        property (it will not equal None after a sucessful run of this method)

        To fit the sklearn Classifier interface, this method return a reduced
        floating metric value for the model.
        """
        devset = self.compute_devset(X, targets)

        eval_model = self._llm
        dspy.configure(lm=eval_model)
        # The evaluator only enable to automate the standard workflow of dspy
        # for running evaluation inferences but this workflow is not suitable
        # for a per-field evaluation
        evaluate = get_evaluator(devset, return_outputs=True)
        print_log("Tracing ready!\n")
        results = cast(
            Tuple[float, List[Tuple[Example, Prediction, float]]],
            evaluate(self._module),
        )
        self._cached_score_results = ResultSchema.validate(
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
        )

        return results[0]
