"""Subpackage for loading all the data.

This cover metatdata tables of Magoh's records, but also the sets of thesauri
related to some fields.
"""

from .load import MagohDataset
from . import thesauri

__all__ = ["MagohDataset", "thesauri"]
