"""Wrapper to get the target Magoh data type from the untrusted output types of
the LLM
"""

from datetime import date
from typing import List, Optional, TypeVar
from .arch_extract_type import (
    ArchaeologicalInterventionContext,
    ArchivalInformation,
    SourceOfInformationInReport,
    TechnicalInformation,
)
from ..target_types import MagohDocumentBuildingData, MagohUniversityData


def process_extensions(ext: Optional[List[str]]):
    if ext is None or len(ext) == 0:
        return None
    return ", ".join(ext)


T = TypeVar("T")


def stringify_if_known[T](e: Optional[T]):
    if e is None:
        return None
    return str(e)


def coalesce_string(d: Optional[str]):
    if d is None:
        return ""
    return d


def coalesce_date(d: Optional[str]):
    if d is None:
        # when the llm has not answered for the date, return today
        # of course, this result will always be wrong, which is
        # what we want, as this information is mandatory and is then wrong
        # if not given
        return str(date.today())
    return d

def normalize_depth(depth: Optional[float]):
    """Apply a simple mathematical operation to be sure the depth is a negative
    float
    We also assume that a reasonable value is between 0.5m to 50m. 50m is not a
    maximum because it is a real-world value. We just use this value because it
    is likely that the llm has missed its unit in outputing such an order of
    magnitude.
    """
    if depth is None:
        return None
    abs_depth = abs(depth)
    if abs_depth >= 50:
        # the output result is likely in cm; convert it into metres
        return abs_depth/100
    return -abs_depth

def to_magoh_university_data(
    context: ArchaeologicalInterventionContext,
    details: TechnicalInformation,
    doc_build_data: SourceOfInformationInReport,
) -> MagohUniversityData:
    return {
        "Sigla": None,  # TODO: figure this out
        "Comune": context.municipality,
        "Ubicazione": context.location,
        "Indirizzo": context.address,
        "Località": context.place,
        "Data intervento": coalesce_date(stringify_if_known(context.intervention_date)),
        "Tipo di intervento": context.intervention_type,
        "Durata": stringify_if_known(context.duration),
        "Eseguito da": stringify_if_known(context.executor),
        "Direzione scientifica": stringify_if_known(context.principal_investigator),
        "Estensione": process_extensions(context.extension),
        "Numero di saggi": details.sample_number,
        "Profondità massima": normalize_depth(details.max_depth),
        "Geologico": details.geology,
        "OGD": details.historical_information_class,
        "OGM": doc_build_data.document_source_type,
        "Profondità falda": normalize_depth(details.groundwater_depth),
    }


def to_magoh_build_data(
    context: ArchaeologicalInterventionContext,
    doc_build_data: SourceOfInformationInReport,
    arch_metadata: Optional[ArchivalInformation],
) -> MagohDocumentBuildingData:
    return {
        "Istituzione": coalesce_string(doc_build_data.institution),
        "Funzionario competente": "".join(
            [str(name) for name in context.on_site_qualified_official]
        )
        if context.on_site_qualified_official is not None
        else "",
        "Tipo di documento": doc_build_data.document_type,
        "Protocollo": arch_metadata.protocol if arch_metadata is not None else "",
        "Data Protocollo": arch_metadata.protocol_date
        if arch_metadata is not None
        else "",
    }
