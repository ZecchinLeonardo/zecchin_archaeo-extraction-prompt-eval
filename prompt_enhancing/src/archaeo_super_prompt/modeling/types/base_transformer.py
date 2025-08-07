"""Base Function Transformer for the pipeline.

Powered by sklearn base class
"""

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for initiate a function transformer with params."""
    def __init__(self) -> None:
        """Neutral init."""
        super().__init__()

    def set_output(self, transform=None):  # type: ignore
        """Function for the support of pandas."""
        # no-op to support sklearn pipeline compatibility
        return self

    def fit(self, X, y):
        """Neutral fit function."""
        X = X  # unused
        y = y  # unused
        return self

    @abstractmethod
    def transform(self, X):
        """Implement the function for transforming the data."""
        raise NotImplementedError
