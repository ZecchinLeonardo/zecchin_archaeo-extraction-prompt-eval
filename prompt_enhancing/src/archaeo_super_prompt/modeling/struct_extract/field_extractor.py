"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from typing import cast, override
import dspy

from archaeo_super_prompt.types.intervention_id import InterventionId

from ...types.featured_chunks import FeaturedChunks

class TypedModule[DInput, DOutput](dspy.Module):
    def typed_forward(self, inpt: DInput) -> DOutput:
        return cast(DOutput, cast(dspy.Prediction, self(**inpt)).to_dict())

class FieldExtractor[DInput, DOutput, DFOutputType](
    ClassifierMixin, BaseEstimator, TransformerMixin, ABC
):
    """Abstract class for extracting one field from featured chunks."""

    def __init__(self, model: TypedModule[DInput, DOutput]) -> None:
        """Initialize the abstract class with the custom dspy module.

        Arguments:
            model: the dspy module which will be used for the training and the \
inference
        """
        self._model = model
        super().__init__()

    @abstractmethod
    @classmethod
    def _filter_chunks(cls, X: FeaturedChunks) -> FeaturedChunks:
        pass

    @abstractmethod
    @classmethod
    def _to_dspy_input(cls, X: FeaturedChunks) -> DInput:
        pass

    @abstractmethod
    @classmethod
    def _compare_values(cls, predicted: DOutput, expected: DOutput):
        pass

    def fit(self, X: FeaturedChunks, y):
        # TODO:
        return self

    def transform(self, X: FeaturedChunks) -> DFOutputType:
        """Generic transform operation
        """
        input = { id_: self.to_dspy_inputs(source) for cast(int, id_), source in self._filter_chunks(X).groupby("id") }

    @override
    def score(self, X: FeaturedChunks, y, sample_weight=None):
        return super().score(X, y, sample_weight)
