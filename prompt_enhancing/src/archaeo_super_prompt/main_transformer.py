from typing import List, Tuple, cast

from dspy import Example, Prediction, dspy

# import mlflow
import pandas
from pandera.typing import DataFrame

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.debug_log import print_log
from archaeo_super_prompt.evaluation.evaluate import get_evaluator
from archaeo_super_prompt.evaluation.load_examples import DevSet
from archaeo_super_prompt.models.main_pipeline import (
    ExtractDataFromInterventionReport,
    ExtractedInterventionData,
)
from archaeo_super_prompt.types.intervention_id import InterventionId
from .output import save_outputs


from .language_model import load_model
from .types.pdfchunks import (
    PDFChunkDataset,
    PDFChunkPerInterventionDataset,
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

    def transform(self, X: PDFChunkDataset) -> DataFrame:
        answers = {
            id_: answer
            for id_, answer in {
                id_: self._module.forward_and_type(
                    cast(PDFChunkPerInterventionDataset, source)
                )
                for id_, source in X.groupby("id")
            }.items()
            if answer is not None
        }
        keys, values = zip(*answers.items())
        answer_df = pandas.json_normalize(cast(list[dict], list(values)))
        answer_df["id"] = cast(List[InterventionId], list(keys))
        return cast(DataFrame, answer_df)

    # def score(self, X: PDFChunkDataset, targets: MagohDataset):
    #     # TODO:
    #     devset: DevSet = [
    #         dspy.Example(
    #             document_ocr_scan=X[id_], answer=targets.get_answer(id_)
    #         ).with_inputs("document_ocr_scan")
    #         for id_ in X
    #     ]
    #     eval_model = self._module.get_lm()
    #     if eval_model is None:
    #         raise Exception("")
    #     dspy.configure(lm=eval_model)
    #     mlflow.set_experiment("Experiment")
    #     mlflow.dspy.autolog(log_evals=True)  # type: ignore
    #     evaluate = get_evaluator(devset, return_outputs=True)
    #     print_log("Tracing ready!\n")
    #     with mlflow.start_run() as active_run:
    #         results = cast(
    #             Tuple[float, List[Tuple[Example, Prediction, float]]],
    #             evaluate(self._module, active_run),
    #         )
    #         save_outputs(
    #             (
    #                 (ex.answer, cast(ExtractedInterventionData, pred.toDict()), score)
    #                 for ex, pred, score in filter(lambda t: t[1].toDict(), results[1])
    #             )
    #         )
    #         return results[0]
