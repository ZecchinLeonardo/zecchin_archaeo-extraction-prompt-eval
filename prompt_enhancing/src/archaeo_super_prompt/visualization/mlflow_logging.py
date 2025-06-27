"""Module with functions to manage the mlflow logging and artifact saving"""

from pathlib import Path
from typing import Tuple, cast
from pandera.typing.pandas import DataFrame
from feature_engine.pipeline import Pipeline
import mlflow
import mlflow.dspy as mldspy

from archaeo_super_prompt.types.intervention_id import InterventionId
from archaeo_super_prompt.types.pdfchunks import PDFChunkDataset
from archaeo_super_prompt.types.pdfpaths import buildPdfPathDataset

from ..main_transformer import MagohDataExtractor

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


def save_models(pipeline: Pipeline, input_example: Tuple[InterventionId, Path]):
    """Save the dspy model for an inspection. The signature is just
    representative. 
    """
    # TODO: log the sklearn pipeline model too
    extractorModel = cast(MagohDataExtractor, pipeline.named_steps["extractor"])
    dspy_model_input_example = {
        "document_ocr_scans__df": extractorModel.compute_model_input(
            cast(
                PDFChunkDataset,
                Pipeline(pipeline.steps[:-1]).transform(
                    buildPdfPathDataset([input_example])
                ),
            )
        )[0][1]
    }
    dspy_model_input_example = dspy_model_input_example # unused for now

    mldspy.log_model(
        extractorModel.dspy_model,
        "dspy_extraction_model",
        # TODO: make the dspy model input (custom class) json serializable
        # input_example=dspy_model_input_example,
    )
    pass
