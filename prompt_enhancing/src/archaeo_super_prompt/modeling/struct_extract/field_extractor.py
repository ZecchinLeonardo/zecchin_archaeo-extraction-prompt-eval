"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from typing import cast, override
from pandera.typing.pandas import DataFrame
import pandas as pd
import dspy

from ..entity_extractor.types import ChunksWithThesaurus
from . import types as extract_input_type


class TypedDspyModule[DInput, DOutput](dspy.Module):
    """A dspy module but with a typed wrapper for the forward function."""

    def typed_forward(self, inpt: DInput) -> DOutput:
        """Carry out a type safe forward on the module."""
        return cast(DOutput, cast(dspy.Prediction, self(**inpt)).to_dict())


class FieldExtractor[DInput, DOutput, SuggestedOutputType, DFOutput](
    ClassifierMixin, BaseEstimator, TransformerMixin, ABC
):
    """Abstract class for extracting one field from featured chunks.

    Genericity:
    As Python does not support a lot of type checking features, the genericity
    constraints are explicited here:
    - DInput is a subtype of TypedDict
    - DOutput is a subtype of TypedDict
    - DFOutputType is a subtype of pandera.pandas.DataFrameModel
    """

    def __init__(
        self,
        model: TypedDspyModule[DInput, DOutput],
        example: tuple[DInput, DOutput],
    ) -> None:
        """Initialize the abstract class with the custom dspy module.

        Arguments:
            model: the dspy module which will be used for the training and the \
inference
            df_schemas: given for type checking
            example: a dspy input-output pair enabling to type check at \
runtime the genericity and also to be able to log the model in mlflow
        """
        super().__init__()
        self._model = model
        # check at initialization at runtime if DInput and DOutput are subtypes
        # of dict
        i, o = example
        if not isinstance(i, dict) or not isinstance(o, dict):
            raise Exception(
                "Type Error: both DInput and DOutput be subtypes of dict"
            )
        self._example = example

    @abstractmethod
    def _to_dspy_input(
        self,
        x: extract_input_type.InputForExtractionRowSchema[
            SuggestedOutputType
        ],
    ) -> DInput:
        """Convert the uniformized extraction input for one intervention into one dict input for the dspy model."""
        pass

    @abstractmethod
    @classmethod
    def _compare_values(cls, predicted: DOutput, expected: DOutput) -> float:
        pass

    def fit(
        self,
        X: DataFrame[
            extract_input_type.InputForExtraction[SuggestedOutputType]
        ],
        y,
    ):
        """Optimize the dspy model according to the given dataset."""
        # TODO:
        return self

    def transform(
        self,
        X: DataFrame[
            extract_input_type.InputForExtraction[SuggestedOutputType]
        ],
    ) -> DataFrame[DFOutput]:
        """Generic transform operation."""
        inputs = {
            row.id: self._to_dspy_input(row)
            for row in extract_input_type.itertuples(X)
        }
        return cast(
            DataFrame[DFOutput],
            pd.DataFrame(
                [
                    {
                        "id": intervention_id,
                        **cast(dict, self._model.typed_forward(inpt)),
                    }
                    for intervention_id, inpt in inputs.items()
                ]
            ),
        )

    @override
    def score(self, X: ChunksWithThesaurus, y, sample_weight=None):
        # TODO: set the local dspy evaluation
        return super().score(X, y, sample_weight)
