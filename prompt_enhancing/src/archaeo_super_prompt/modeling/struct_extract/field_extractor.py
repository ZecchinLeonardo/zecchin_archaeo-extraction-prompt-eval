"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from typing import cast, override
from pandera.typing.pandas import DataFrame
import pandas as pd
import dspy

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.modeling.struct_extract.evaluation.evaluate import (
    get_evaluator,
)
from archaeo_super_prompt.types.intervention_id import InterventionId

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
        llm_model: dspy.LM,
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
        self.__llm_model = llm_model
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
        x: extract_input_type.InputForExtractionRowSchema[SuggestedOutputType],
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

    @abstractmethod
    @classmethod
    def select_answer(cls, y: MagohDataset, id: InterventionId) -> DOutput:
        pass

    def compute_devset(
        self,
        X: DataFrame[
            extract_input_type.InputForExtraction[SuggestedOutputType]
        ],
        y: MagohDataset,
    ):
        inputs = {
            row.id: self._to_dspy_input(row)
            for row in extract_input_type.itertuples(X)
        }
        return [
            dspy.Example(
                **model_input,
                # TODO: select only one field with an abstract
                **self.select_answer(y, InterventionId(cast(int, id_))),
            ).with_inputs(*cast(dict, model_input).keys())
            for id_, model_input in inputs.items()
        ]

    @override
    def score(
        self,
        X: DataFrame[
            extract_input_type.InputForExtraction[SuggestedOutputType]
        ],
        y: MagohDataset,
        sample_weight=None,
    ):
        """Run a local evaluation of the dpsy model over the given X dataset.

        Also save the per-field results for each test record in a cached
        dataframe, accessible after the function call with the score_results
        property (it will not equal None after a sucessful run of this method)

        To fit the sklearn Classifier interface, this method return a reduced
        floating metric value for the model.
        """
        sample_weight = sample_weight  # unused

        devset = self.compute_devset(X, y)

        eval_model = self.__llm_model
        dspy.configure(lm=eval_model)
        evaluate = get_evaluator(devset, return_outputs=True)
        results = cast(
            tuple[float, list[tuple[dspy.Example, dspy.Prediction, float]]],
            evaluate(self._model),
        )
        return results[0]
