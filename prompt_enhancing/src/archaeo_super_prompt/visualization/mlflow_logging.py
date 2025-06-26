"""Module with functions to manage the mlflow logging and artifact saving"""

from pandera.typing.pandas import DataFrame
import mlflow

from ..types.results import ResultSchema
from .prettify_field_names import prettify_field_names


def save_table_in_artifacts(score_results: DataFrame[ResultSchema]):
    for fieldName, resultPerField in prettify_field_names(score_results).groupby(
        "field_name"
    ):
        mlflow.log_table(
            resultPerField.drop(columns=["field_name", "evaluation_method"]),
            f"eval_{fieldName}.json",
        )


def save_metric_scores(
    reduced_dspy_eval_score: float, score_results: DataFrame[ResultSchema]
):
    mlflow.log_metric("reduced_dspy_eval_score", reduced_dspy_eval_score)
    for fieldName, resultPerField in prettify_field_names(score_results).groupby(
        "field_name"
    ):
        mlflow.log_metric(str(fieldName), resultPerField["metric_value"].mean())
