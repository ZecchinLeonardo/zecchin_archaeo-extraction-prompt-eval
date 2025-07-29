"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from logging import warning
from typing import cast, override
from pandera.typing.pandas import DataFrame
import pandas as pd
import dspy
import tqdm

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.types.intervention_id import InterventionId
from archaeo_super_prompt.types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)

from . import types as extract_input_type
from ..types.detailed_evaluator import DetailedEvaluatorMixin


class TypedDspyModule[DInput, DOutput](dspy.Module):
    """A dspy module but with a typed wrapper for the forward function."""

    def typed_forward(self, inpt: DInput) -> DOutput:
        """Carry out a type safe forward on the module."""
        return cast(dspy.Prediction, self(**inpt)).toDict()  # type: ignore


class FieldExtractor[
    DSPyInput,
    DSPyOutput,
    InputDataFrameWithKnowledge: extract_input_type.BaseInputForExtraction,
    InputDataFrameWithKnowledgeRowSchema: extract_input_type.BaseInputForExtractionRowSchema,
    DFOutput: BasePerInterventionFeatureSchema,
](
    DetailedEvaluatorMixin[
        DataFrame[InputDataFrameWithKnowledge],
        MagohDataset,
    ],
    ABC,
):
    """Abstract class for extracting one field from featured chunks.

    Genericity:
    As Python does not support a lot of type checking features, the genericity
    constraints are explicited here:
    - DInput is a subtype of TypedDict, whose keys bring semantics used by \
the DSPy model as input in its forward method.
    - DOutput is a subtype of TypedDict
    - DFOutputType is a subtype of pandera.pandas.DataFrameModel
    """

    def __init__(
        self,
        llm_model: dspy.LM,
        model: TypedDspyModule[DSPyInput, DSPyOutput],
        example: tuple[DSPyInput, DSPyOutput],
        optimized: TypedDspyModule[DSPyInput, DSPyOutput] | None = None,
    ) -> None:
        """Initialize the abstract class with the custom dspy module.

        Arguments:
            llm_model: the dspy chat lm to be used for the extraction 
            model: the dspy module which will be used for the training and the \
inference
            df_schemas: given for type checking
            example: a dspy input-output pair enabling to type check at \
runtime the genericity and also to be able to log the model in mlflow
            optimized: the already trained prompt model, if existing
        """
        super().__init__()
        self.__llm_model = llm_model
        self._prompt_model = model
        self._optimized_prompt_model: (
            TypedDspyModule[DSPyInput, DSPyOutput] | None
        ) = optimized
        # check at initialization at runtime if DInput and DOutput are subtypes
        # of dict
        i, o = example
        if not isinstance(i, dict) or not isinstance(o, dict):
            raise Exception(
                "Type Error: both DInput and DOutput be subtypes of dict"
            )
        self._example = example

    @classmethod
    def _itertuples(cls, X: DataFrame[InputDataFrameWithKnowledge]):
        return cast(
            Iterator[InputDataFrameWithKnowledgeRowSchema], X.itertuples()
        )

    @abstractmethod
    def _to_dspy_input(
        self,
        x: InputDataFrameWithKnowledgeRowSchema,
    ) -> DSPyInput:
        """Convert the uniformized extraction input for one intervention into one dict input for the dspy model."""
        raise NotImplementedError

    def _identity_output_set_transform_to_df(
        self, y: Iterator[tuple[InterventionId, DSPyOutput]]
    ) -> pd.DataFrame:
        """Method to directly transform the set of dspy output into a dataframe.

        Use it if needed in the transform_dspy_output implementation. For a
        type-safe usage, in your implementation, pass the output of this
        method in a scheme validation function.
        """
        return pd.DataFrame(
            [
                {
                    "id": intervention_id,
                    **cast(dict, dspy_output),
                }
                for intervention_id, dspy_output in y
            ]
        )

    @abstractmethod
    def _transform_dspy_output(
        self, y: Iterator[tuple[InterventionId, DSPyOutput]]
    ) -> DataFrame[DFOutput]:
        """Transform the map of outputs into an output DataFrame with the wanted schema.

        If you want to directly use the attributes of the dspy dict output into
        the dataframe, use the _identity_output_set_transform_to_df method and
        validate this output from your DataFrameModel.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _compare_values(
        cls, predicted: DSPyOutput, expected: DSPyOutput
    ) -> tuple[float, float]:
        """Compute a metric to compare the expected output with the predicted one.

        Return:
            a score between 0 and 1
            a treshold score above which the comparison is considered as successful
        """
        raise NotImplementedError

    @override
    def fit(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
        y: MagohDataset,
    ):
        """Optimize the dspy model according to the given dataset."""
        if self._optimized_prompt_model is not None:
            return self
        with dspy.settings.context(lm=self.__llm_model):
            tp = dspy.MIPROv2(
                metric=self._dspy_metric, auto="medium", num_threads=24
            )
            self._optimized_prompt_model = cast(
                TypedDspyModule[DSPyInput, DSPyOutput],
                tp.compile(
                    self._prompt_model,
                    trainset=self._compute_devset(X, y),
                    max_bootstrapped_demos=2,
                    max_labeled_demos=2,
                ),
            )
        return self

    @override
    def transform(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
    ) -> DataFrame[DFOutput]:
        """Generic transform operation."""
        inputs = (
            (InterventionId(row.id), self._to_dspy_input(row))
            for row in self._itertuples(X)
        )
        with dspy.settings.context(lm=self.__llm_model):
            return self._transform_dspy_output(
                (
                    intervention_id,
                    self._prompt_model.typed_forward(inpt),
                )
                for intervention_id, inpt in tqdm.tqdm(
                    inputs,
                    total=len(X),
                    desc="Field extraction",
                    unit="processed intervention",
                )
            )

    @classmethod
    @abstractmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        """Among the given set of intervention records, select only those with suitable answers for a training or an evaluation."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DSPyOutput]:
        raise NotImplementedError

    def _compute_devset(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
        y: MagohDataset,
    ):
        good_ids = self.filter_training_dataset(y, set(InterventionId(id_) for id_ in X["id"].to_list()))
        not_good_ids = X[~(X["id"].isin(good_ids))]["id"].to_list()
        if not_good_ids:
            warning(f"These records will not be used in the devset, as their answers are incorrect: {not_good_ids}")
        inputs = {
            InterventionId(row.id): self._to_dspy_input(row)
            for row in self._itertuples(X[X["id"].isin(good_ids)])
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
            cast(DSPyOutput, prediction.toDict()),
            cast(DSPyOutput, example.toDict()),
        )
        if trace is None:
            return result
        return result >= passable_treshold

    @override
    def score(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
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
            score = cast(float, evaluator(self._prompt_model))
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
            evaluator(self._prompt_model),
        )
        # TODO: return a df after the score
        return results
