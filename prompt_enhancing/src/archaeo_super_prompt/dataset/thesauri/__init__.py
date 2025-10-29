"""Code for loading thesaurus sets from data files."""

from .comune_province import load_comune, load_comune_with_provincie, ComuneProvincia, Provincia
from .esecutore import load_esecutore_candidates
from .protocollo import load_protocollo_candidates
from .tipo import load_tipo_candidates
from .ogd import load_ogd_candidates
from .luogo import load_luogo_candidates
from .funzionario import load_funzionario_candidates

__all__ = ["load_comune", "load_comune_with_provincie", "ComuneProvincia",
           "Provincia", "load_esecutore_candidates", "load_protocollo_candidates", "load_tipo_candidates", "load_ogd_candidates", "load_luogo_candidates", "load_funzionario_candidates"]

