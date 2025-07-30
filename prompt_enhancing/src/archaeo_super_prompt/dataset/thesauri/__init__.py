"""Code for loading thesaurus sets from data files."""

from .comune_province import load_comune, load_comune_with_provincie, ComuneProvincia, Provincia

__all__ = ["load_comune", "load_comune_with_provincie", "ComuneProvincia",
           "Provincia"]

