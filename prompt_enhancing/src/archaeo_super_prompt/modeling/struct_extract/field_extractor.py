"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from typing import cast, override
from pandera.typing.pandas import DataFrame
import pandas as pd
import dspy

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.types.intervention_id import InterventionId

from . import types as extract_input_type
from ..types.detailed_evaluator import DetailedEvaluatorMixin


class TypedDspyModule[DInput, DOutput](dspy.Module):
    """A dspy module but with a typed wrapper for the forward function."""

    def typed_forward(self, inpt: DInput) -> DOutput:
        """Carry out a type safe forward on the module."""
        return cast(DOutput, cast(dspy.Prediction, self(**inpt)).to_dict())


class FieldExtractor[DInput, DOutput, SuggestedOutputType, DFOutput](
    DetailedEvaluatorMixin[
        DataFrame[extract_input_type.InputForExtraction[SuggestedOutputType]],
        MagohDataset,
    ],
    ABC,
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
            llm_model: the dspy chat lm to be used for the extraction 
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
    def _compare_values(
        cls, predicted: DOutput, expected: DOutput
    ) -> tuple[float, float]:
        """Compute a metric to compare the expected output with the predicted one.

        Return:
            a score between 0 and 1
            a treshold score above which the comparison is considered as successful
        """
        pass

    @override
    def fit(
        self,
        X: DataFrame[
            extract_input_type.InputForExtraction[SuggestedOutputType]
        ],
        y: MagohDataset,
    ):
        """Optimize the dspy model according to the given dataset."""
        # TODO:
        X = X  # unused
        y = y  # unused
        return self

    @override
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
        with dspy.settings.context(lm=self.__llm_model):
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
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DOutput]:
        pass

    def _compute_devset(
        self,
        X: DataFrame[
            extract_input_type.InputForExtraction[SuggestedOutputType]
        ],
        y: MagohDataset,
    ):
        inputs = {
            InterventionId(cast(int, row.id)): self._to_dspy_input(row)
            for row in extract_input_type.itertuples(X)
        }
        answers = self._select_answers(y, set(inputs.keys()))
        return [
            dspy.Example(
                **model_input,
                # TODO: select only one field with an abstract
                **answers[id_],
            ).with_inputs(*cast(dict, model_input).keys())
            for id_, model_input in inputs.items()
        ]

    @classmethod
    def _dspy_metric(
        cls, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> float | bool:
        result, passable_treshold = cls._compare_values(
            cast(DOutput, prediction.toDict()), cast(DOutput, example.toDict())
        )
        if trace is None:
            return result
        return result >= passable_treshold

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

        devset = self._compute_devset(X, y)

        with dspy.settings.context(lm=self.__llm_model):
            evaluator = dspy.Evaluate(
                devset=devset,
                metric=self._dspy_metric,
                return_outputs=False,
                provide_traceback=True,  # TODO: remove it for traceback
                num_threads=1,  # TODO: set it
                display_progress=True,
                display_table=5,
            )
            score = cast(float, evaluator(self._model))
        return score

    @override
    def score_and_transform(self, X, y):
        devset = self._compute_devset(X, y)
        evaluator = dspy.Evaluate(
            devset=devset,
            metric=self._dspy_metric,
            return_outputs=True,
            provide_traceback=True,  # TODO: remove it for traceback
            num_threads=1,  # TODO: set it
            display_progress=True,
            display_table=5,
        )
        results = cast(
            tuple[float, list[tuple[dspy.Example, dspy.Prediction, float]]],
            evaluator(self._model),
        )
        # TODO: return a df after the score
        return results
