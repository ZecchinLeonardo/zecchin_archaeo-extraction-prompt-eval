"""Module with functions to manage the mlflow logging and artifact saving."""

import mlflow
import mlflow.dspy as mldspy
from pandera.typing.pandas import DataFrame

from archaeo_super_prompt.modeling.struct_extract.field_extractor import (
    FieldExtractor,
)

from ..types.results import ResultSchema
from .prettify_field_names import prettify_field_names


def save_table_in_artifacts(score_results: DataFrame[ResultSchema]):
    """Save the detailed results."""
    for fieldName, resultPerField in prettify_field_names(
        score_results
    ).groupby("field_name"):
        mlflow.log_table(
            resultPerField.drop(columns=["field_name", "evaluation_method"]),
            f"eval_{fieldName}.json",
        )


def save_metric_scores(
    reduced_dspy_eval_score: float, score_results: DataFrame[ResultSchema]
):
    """Save the per-field metric scores from the global results."""
    mlflow.log_metric("reduced_dspy_eval_score", reduced_dspy_eval_score)
    for fieldName, resultPerField in prettify_field_names(
        score_results
    ).groupby("field_name"):
        mlflow.log_metric(
            str(fieldName), resultPerField["metric_value"].mean()
        )


def save_models(extractorModel: FieldExtractor):
    """Save the dspy model for an inspection."""
    # TODO: log the sklearn pipeline model too
    mldspy.log_model(
        extractorModel.prompt_model_,
        extractorModel.field_to_be_extracted(),
        # WARN: no signature is inferrable for now in cause of the usage of
        # pydantic models which is not supported by the current dspy
        # integration
    )
