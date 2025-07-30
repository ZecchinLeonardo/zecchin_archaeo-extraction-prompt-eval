"""Subpackage for loading all the data.

This cover metadata tables of Magoh's records, but also the sets of thesauri
related to some fields.
"""

from .load import MagohDataset, SamplingParams, IdSet
from . import thesauri

__all__ = ["MagohDataset", "SamplingParams", "IdSet", "thesauri"]
