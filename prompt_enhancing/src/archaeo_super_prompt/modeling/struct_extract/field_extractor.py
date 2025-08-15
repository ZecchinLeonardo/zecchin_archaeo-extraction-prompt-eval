"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from logging import warning
from pathlib import Path
from typing import Literal, cast, override
from pydantic import BaseModel
from pandera.typing.pandas import DataFrame
import pandas as pd
import dspy
import tqdm

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.types.intervention_id import InterventionId
from archaeo_super_prompt.types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...types.results import ResultSchema

from . import types as extract_input_type
from ..types.detailed_evaluator import DetailedEvaluatorMixin

from . import language_model as lm_provider_mod


EvalDetailedResult = list[tuple[dspy.Example, dspy.Prediction, float]]
LLMProvider = Literal["vllm", "ollama", "openai"]


def to_prediction(output: BaseModel) -> dspy.Prediction:
    """Call this function with the pydantic-typed output for return in forward."""
    return dspy.Prediction(**output.model_dump())


def prediction_to_output[DSPyOutput](
    output_constructor: type[DSPyOutput], pred: dspy.Prediction
) -> DSPyOutput:
    """Inverse of the method above.

    Expect the prediction to be built from the _to_prediction method above
    """
    return output_constructor(**pred.toDict())


class FieldExtractor[
    DSPyInput: BaseModel,
    DSPyOutput: BaseModel,
    InputDataFrameWithKnowledge: extract_input_type.BaseInputForExtraction,
    InputDataFrameWithKnowledgeRowSchema: extract_input_type.BaseInputForExtractionRowSchema,
    DFOutput: BasePerInterventionFeatureSchema,
](
    DetailedEvaluatorMixin[
        DataFrame[InputDataFrameWithKnowledge],
        MagohDataset,
        DataFrame[ResultSchema],
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
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
        model: dspy.Module,
        example: tuple[DSPyInput, DSPyOutput],
        output_constructor: type[DSPyOutput],
    ) -> None:
        """Initialize the abstract class with the custom dspy module.

        Arguments:
            llm_model_provider: the service from which the llm must be fetched
            llm_model_id: the dspy chat lm to be used for the extraction 
            llm_temperature: the temperature of the llm during the prompts of \
this model
            model: the dspy module which will be used for the training and the \
inference
            example: a dspy input-output pair enabling to type check at \
runtime the genericity and also to be able to log the model in mlflow
            output_constructor: the type of the output model for building it \
generically from dictionnary expansion

        Environment variables:
            According to the llm provider, either the following env vars is
            required:
               OPENAI_API_KEY
               OLLAMA_SERVER_BASE_URL (default to http://localhost:11434)
               VLLM_SERVER_BASE_URL (default to http://localhost:8006/v1)
        """
        super().__init__()
        self.llm_model_provider: LLMProvider = llm_model_provider
        self.llm_model_id = llm_model_id
        self.llm_temperature = llm_temperature
        self._base_dspy_module = model
        self._example = example
        self._output_constructor = output_constructor

    def _infer_language_model(self):
        match self.llm_model_provider:
            case "ollama":
                return lm_provider_mod.get_ollama_model(
                    self.llm_model_id, self.llm_temperature
                )
            case "vllm":
                return lm_provider_mod.get_vllm_model(
                    self.llm_model_id, self.llm_temperature
                )
            case "openai":
                return lm_provider_mod.get_openai_model(
                    self.llm_model_id, self.llm_temperature
                )

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
                    **dspy_output.model_dump(),
                }
                for intervention_id, dspy_output in y
            ]
        ).set_index("id")

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
        *,
        compiled_dspy_model_path: Path | None = None,
        skip_optimization=False,
        **kwargs,
    ):
        """Optimize the dspy model according to the given dataset.

        Arguments:
           X: the input dataframe with the required fields for the FieldExtractor
           y: the Magoh training dataset
           compiled_dspy_model_path: if given, a path to an already optimized dspy model, so this prompt model is directly used without reoptimize the program
           skip_optimization: if set to True, then the model is fitted with the not optimized dspy program
           kwargs: nothing usefull (just to fit the initial overriding)
        """
        kwargs = kwargs  # unused
        if skip_optimization:
            self.prompt_model_ = self._base_dspy_module
            return self
        if compiled_dspy_model_path is not None:
            self._base_dspy_module.load(compiled_dspy_model_path)
            self.prompt_model_ = self._base_dspy_module
            return self
        with dspy.settings.context(lm=self._infer_language_model()):
            tp = dspy.MIPROv2(
                metric=self._dspy_metric, auto="medium", num_threads=24
            )
            self.prompt_model_ = tp.compile(
                self._base_dspy_module,
                trainset=list(self._compute_devset(X, y)[1]),
                max_bootstrapped_demos=2,
                max_labeled_demos=2,
                requires_permission_to_run=False,
            )
            return self

    def _typed_forward(self, inpt: DSPyInput) -> DSPyOutput:
        """Carry out a type safe forward on the dspy module."""
        return prediction_to_output(
            self._output_constructor,
            cast(dspy.Prediction, self.prompt_model_(**inpt.model_dump())),
        )

    @override
    def predict(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
    ) -> DataFrame[DFOutput]:
        """Generic transform operation."""
        inputs = (
            (InterventionId(row.Index), self._to_dspy_input(row))
            for row in self._itertuples(X)
        )
        with dspy.settings.context(lm=self._infer_language_model()):
            return self._transform_dspy_output(
                (
                    intervention_id,
                    self._typed_forward(inpt),
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
    ) -> tuple[tuple[int, ...], tuple[dspy.Example, ...]]:
        good_ids = self.filter_training_dataset(
            y, set(InterventionId(id_) for id_ in list(X.index))
        )
        not_good_ids = X[~(X.index.isin(good_ids))].index.to_list()
        if not_good_ids:
            warning(
                f"These records will not be used in the devset, as their answers are incorrect: {not_good_ids}"
            )
        inputs = {
            InterventionId(row.Index): self._to_dspy_input(row)
            for row in self._itertuples(X[X.index.isin(good_ids)])
        }
        answers = self._select_answers(y, set(inputs.keys()))
        kept_ids, examples = zip(
            *(
                (
                    id_,
                    (
                        lambda model_input: dspy.Example(
                            **model_input,
                            # TODO: select only one field with an abstract
                            **answers[id_].model_dump(),
                        ).with_inputs(*model_input.keys())
                    )(inputs[id_].model_dump()),
                )
                for id_ in inputs.keys()
            )
        )
        return kept_ids, examples

    def _dspy_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> float | bool:
        result, passable_treshold = self._compare_values(
            self._output_constructor(**prediction.toDict()),
            self._output_constructor(**example.toDict()),
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

        _, devset = self._compute_devset(X, y)

        with dspy.settings.context(lm=self._infer_language_model()):
            evaluator = dspy.Evaluate(
                devset=list(devset[1]),
                metric=self._dspy_metric,
                return_outputs=False,
                provide_traceback=True,  # TODO: remove it for traceback
                num_threads=1,  # TODO: set it
                display_progress=True,
                display_table=5,
            )
            score = cast(float, evaluator(self.prompt_model_))
        return score

    @override
    def score_and_transform(self, X, y):
        kept_ids, devset = self._compute_devset(X, y)
        with dspy.settings.context(lm=self._infer_language_model()):
            evaluator = dspy.Evaluate(
                devset=list(devset),
                metric=self._dspy_metric,
                return_outputs=True,
                provide_traceback=True,  # TODO: remove it for traceback
                num_threads=1,  # TODO: set it
                display_progress=True,
                display_table=5,
            )
            score, score_table = cast(
                tuple[float, EvalDetailedResult],
                evaluator(self.prompt_model_),
            )
            return score, ResultSchema.validate(
                pd.DataFrame(
                    [
                        {
                            "id": id_,
                            "field_name": self.field_to_be_extracted(),
                            "metric_value": score,
                            # TODO: specify the evaluation method
                            "evaluation_method": "not specified yet",
                            "expected_value": {
                                k: ex_dict[k] for k in pred_dict
                            },
                            "predicted_value": pred_dict,
                        }
                        for id_, (ex_dict, pred_dict, score) in zip(
                            kept_ids,
                            (
                                (ex.toDict(), pred.toDict(), score)
                                for ex, pred, score in score_table
                            ),
                        )
                    ]
                ),
                lazy=True,
            )

    @staticmethod
    @abstractmethod
    def field_to_be_extracted() -> str:
        """A human label/description of the field related to the Extractor."""
        raise NotImplementedError

    @property
    def signature_example(self):
        """Return an example of input/output dict pair for the dspy model.

        This property is usefull for a logging by mlflow.
        """
        return self._example

    @property
    def lm(self):
        """Return the llm model."""
        return self._infer_language_model()
