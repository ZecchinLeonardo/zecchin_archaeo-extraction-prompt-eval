"""Base class for all the pipeline Classifiers used in this software.

To evaluate each classifying layer of the pipeline in this DAG model, we want
each Transformer to be implement the following methods:
- the standard fit and transform methods
- the score method to evaluate the accuracy (local or independent) of the layer
- the score_and_transform to score and also get gathered results for visualization
"""

from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin, BaseEstimator
from typing import Any, override


class DetailedEvaluatorMixin[InputBatch, TargetDataSet, EvaluationDetail](
    ClassifierMixin, BaseEstimator, ABC
):
    """A model which can be evaluated and must be trained."""

    def fit(self, X: InputBatch, y: TargetDataSet, **kwargs):
        """Train the model with the given targey data.

        Override it to implement a training.
        """
        kwargs = kwargs  # unused
        X = X  # unused
        y = y  # unused
        return self

    @abstractmethod
    def predict(self, X: InputBatch) -> Any:
        """The transform method to be implemented."""
        raise NotImplementedError

    @abstractmethod
    @override
    def score(
        self,
        X: InputBatch,
        y: TargetDataSet,
        sample_weight=None,
    ) -> float:
        """The evaluation to compute a local (or independent) accuracy of the model."""
        raise NotImplementedError

    @abstractmethod
    def score_and_transform(
        self, X: InputBatch, y: TargetDataSet
    ) -> tuple[float, EvaluationDetail]:
        """Run an evaluation and return the score with the detailed results.

        Call this method to score the model with getting the detailed results
        for plotting.

        Arguments:
            X: the input batch
            y: the dataset with the target data related to the inputs 

        Return:
            a floating global score
            the detailed results, including the predicted output, the target \
output and the metric value for each comparison.
        """
        raise NotImplementedError
