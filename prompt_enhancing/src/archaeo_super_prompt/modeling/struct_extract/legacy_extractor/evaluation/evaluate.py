import dspy

import mlflow
from pandas import DataFrame


from .compare import validate_magoh_data, reduce_magoh_data_eval

from .load_examples import DevSet


class MyRun:
    """Utils class for stacking the dataframes computed in an evaluation."""

    active_run: mlflow.ActiveRun | None = None
    dataframes: list[tuple[int, dict[str, DataFrame]]] = []

    def __init__(self, active_run: mlflow.ActiveRun | None) -> None:
        self.active_run = active_run


def measure_and_plot(active_run: mlflow.ActiveRun | None):
    """Return an evaluation metric, suffixed by a taks of
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
        metric, _ = measure_and_plot(None)
        results = evaluator(program, metric=metric)
        return results

    return evaluate
