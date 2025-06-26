from typing import Dict, List, Optional, Tuple
import dspy

import mlflow
from pandas import DataFrame


from .compare import validate_magoh_data, reduce_magoh_data_eval

from .load_examples import DevSet


class MyRun:
    """Utils class for stacking the dataframes computed in an evaluation."""

    active_run: Optional[mlflow.ActiveRun] = None
    dataframes: List[Tuple[int, Dict[str, DataFrame]]] = []

    def __init__(self, active_run: Optional[mlflow.ActiveRun]) -> None:
        self.active_run = active_run


def measure_and_plot(active_run: Optional[mlflow.ActiveRun]):
    """
    Return an evaluation metric, suffixed by a taks of
    plotting when all the batch has been processed
    """
    run = MyRun(active_run=active_run)

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None):
        nonlocal run
        result = validate_magoh_data(example, pred, trace)
        # TODO: remanage that later
        # run.dataframes.append(
        #     save_visual_score_table(
        #         example.answer,
        #         cast(ExtractedInterventionData, pred.toDict()),
        #         result,
        #         run.active_run,
        #     )
        # )
        return reduce_magoh_data_eval(result, trace)

    return metric, run


def get_evaluator(devset: DevSet, return_outputs=False):
    # TODO: parametrize some settings
    evaluator = dspy.Evaluate(
        devset=devset,
        return_outputs=return_outputs,
        provide_traceback=True,  # TODO: remove it for traceback
        num_threads=1,
        display_progress=True,
        display_table=5,
    )

    def evaluate(program: dspy.Module):
        metric, run = measure_and_plot(None)
        lm = program.get_lm()
        if lm is None:
            raise Exception(
                "Cannot evaluate without a set language model for the given module"
            )
        # temperature = int(lm.kwargs["temperature"] * 1000)
        # TODO: put the model logging in another place
        # mlflow.dspy.log_model(dspy_model=program, artifact_path=str(Path(f"./outputs/model_temp__{temperature}_x1000")), input_example=devset[0].inputs().toDict()) # type: ignore
        results = evaluator(program, metric=metric)
        return results

    return evaluate
