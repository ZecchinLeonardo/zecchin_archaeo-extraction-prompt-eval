from typing import List, Tuple, cast

from dspy import Example, Prediction, dspy

# import mlflow
import pandas

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.debug_log import print_log
from archaeo_super_prompt.evaluation.evaluate import get_evaluator
from archaeo_super_prompt.evaluation.load_examples import DevSet
from archaeo_super_prompt.models.main_pipeline import (
    ExtractDataFromInterventionReport,
)
from archaeo_super_prompt.target_types import MagohData
from archaeo_super_prompt.types.intervention_id import InterventionId


from .language_model import load_model
from .types.pdfchunks import (
    PDFChunkDataset,
    PDFChunkPerInterventionDataset,
    PDFChunkSetPerInterventionSchema,
)


class MagohDataExtractor:
    def __init__(self, llm_temp=0.0) -> None:
        self._module = ExtractDataFromInterventionReport()
        self._llm = load_model(llm_temp)
        self._module.set_lm(self._llm)

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
                id_: self._module.forward_and_type(
                    PDFChunkPerInterventionDataset(
                        PDFChunkSetPerInterventionSchema.validate(source, lazy=True)
                    )
                )
                for id_, source in X.groupby("id")
            }.items()
            if answer is not None
        }
        ids, output_structured_data = zip(*answers.items())
        answer_df = pandas.json_normalize(
            cast(list[dict], list(output_structured_data))
        )
        answer_df["id"] = cast(List[InterventionId], list(ids))
        return answer_df

    def score(self, X: PDFChunkDataset, targets: MagohDataset):
        """For now, run an evaluation for each field
        """
        def to_dict(answer: MagohData):
            return { "university": answer["university"], "building":
                    answer["building"] }

        devset: DevSet = [
            dspy.Example(
                document_ocr_scans__df=PDFChunkPerInterventionDataset(
                    PDFChunkSetPerInterventionSchema.validate(source, lazy=True)
                ),
                **to_dict(targets.get_answer(cast(int, id_))),
            ).with_inputs("document_ocr_scans__df")
            for id_, source in X.groupby("id")
        ]
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
        # return results[0]
        return results
