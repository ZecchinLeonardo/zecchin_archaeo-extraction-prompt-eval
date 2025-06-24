from pathlib import Path
from typing import Dict, List, Tuple, cast

from dspy import Example, Prediction, dspy
import mlflow
from pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.debug_log import print_log
from archaeo_super_prompt.evaluation.evaluate import get_evaluator
from archaeo_super_prompt.evaluation.load_examples import DevSet
from archaeo_super_prompt.models.main_pipeline import (
    ExtractDataFromInterventionReport,
    ExtractedInterventionData,
)
from archaeo_super_prompt.output import save_outputs
from archaeo_super_prompt.pdf_to_text import OCR_Transformer, TextExtractor

from feature_engine.pipeline import Pipeline

from archaeo_super_prompt.language_model import load_model
from archaeo_super_prompt.signatures.input import PDFSources
from archaeo_super_prompt.types.intervention_id import InterventionId

class MagohDataExtractor:
    def __init__(self, llm_temp=0.0) -> None:
        self._module = ExtractDataFromInterventionReport()
        self._llm = load_model(llm_temp)
        self._module.set_lm(self._llm)

    def fit(self, X: Dict[InterventionId, PDFSources], Y: MagohDataset):
        # TODO: call a dspy optimizer
        X = X
        Y = Y
        return self

    def transform(self, X: Dict[InterventionId, PDFSources]):
        return {id_: self._module.forward_and_type(X[id_]) for id_ in X}

    def score(self, X: Dict[InterventionId, PDFSources], targets: MagohDataset):
        # TODO: pass
        devset: DevSet = [
            dspy.Example(
                document_ocr_scan=X[id_], answer=targets.get_answer(id_)
            ).with_inputs("document_ocr_scan")
            for id_ in X
        ]
        eval_model = self._module.get_lm()
        if eval_model is None:
            raise Exception("")
        dspy.configure(lm=eval_model)
        mlflow.set_experiment("Experiment")
        mlflow.dspy.autolog(log_evals=True)  # type: ignore
        evaluate = get_evaluator(devset, return_outputs=True)
        print_log("Tracing ready!\n")
        with mlflow.start_run() as active_run:
            results = cast(
                Tuple[float, List[Tuple[Example, Prediction, float]]],
                evaluate(self._module, active_run),
            )
            save_outputs(
                (
                    (ex.answer, cast(ExtractedInterventionData, pred.toDict()), score)
                    for ex, pred, score in filter(lambda t: t[1].toDict(), results[1])
                )
            )
            return results[0]


def inputs_from_dataset(dataset: MagohDataset) -> Dict[InterventionId, List[Path]]:
    return {
        id_: cast(
            Series,
            dataset.files[dataset.files["scheda_intervento.id"] == id_]["filepath"],
        ).to_list()
        for id_ in cast(Series, dataset.intervention_data["scheda_intervento.id"])
    }


# TODO: better manage the cache
pipeline = Pipeline(
    [
        ("ocr", OCR_Transformer()),
        ("pdf_reader", TextExtractor()),
        # ("extractor", MagohDataExtractor()),
    ]
)
