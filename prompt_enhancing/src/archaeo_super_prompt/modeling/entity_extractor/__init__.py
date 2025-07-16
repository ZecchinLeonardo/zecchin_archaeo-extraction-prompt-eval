"""Root of the module for infering in the NER model.

The purpose of this model is to extract hints about chunks for helping the
final LLM model to extract some named values for some fields.
"""

from .ner_transformer import NerModel
from .ne_selector import NeSelector
from .types import ChunksWithThesaurus, NamedEntityField

__all__ = ["NerModel", "NeSelector", "ChunksWithThesaurus", "NamedEntityField"]
