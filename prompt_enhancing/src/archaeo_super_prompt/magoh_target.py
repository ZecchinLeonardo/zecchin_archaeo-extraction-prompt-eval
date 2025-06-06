from typing import List, Optional, TypedDict

from .models.main_pipeline import ExtractedInterventionData

MagohUniversityData = TypedDict("MagohUniversityData", {
    "Sigla": Optional[str],
    "Comune": str,
    "Ubicazione": str,
    "Indirizzo": Optional[str],
    "Località": Optional[str],
    "Data intervento": str,
    "Tipo di intervento": str,
    "Durata": Optional[int],
    "Eseguito da": Optional[str],
    "Direzione scientifica": Optional[str],
    "Estensione": Optional[str],
    "Numero di saggi": int, # unsigned
    "Profondità massima": Optional[float], # absolute value but negative
    "Geologico": Optional[bool],
    "OGD":str,
    "OGM": str,
    "Profondità falda": Optional[float]
})

# TODO: MagohCheckedEras

MagohDocumentBuildingData = TypedDict("MagohDocumentBuildingData", {
    "Istituzione": str,
    "Funzionario competente": str,
    "Tipo di documento": str,
    "Protocollo": str,
    "Data Protocollo": str
})

MagohFindingScheme = TypedDict(
    "MagohFindingScheme",
    {
        "I Livello": Optional[str],
        "II Livello": Optional[str],
        "III Livello": Optional[str],
        "Datazione": Optional[int],
        "Datazione Finale": Optional[int],
    },
)

MagohArtificialRecordData = TypedDict(
    "MagohArtificialRecordData", { "id": int }
)

# TODO: add findings to MagohData
MagohData = TypedDict(
    "MagohData",
    {
        "university": MagohUniversityData,
        "building": MagohDocumentBuildingData,
        "scheda-intervento": MagohArtificialRecordData
    }
)

def coalesce_str(elt: Optional[str]):
    return "" if elt is None else elt

def process_extensions(ext: Optional[List[str]]):
    if ext is None:
        return ""
    return ", ".join(ext)

def dID_objects_processing(raw: List[str]):
    if len(raw) == 0:
        return ""
    else:
        return f"Sì ({', '.join(raw)})"

def toMagohData(output: ExtractedInterventionData) -> MagohData:
    context = output["context"]
    details = output["technical_achievements"]
    doc_build_data = output["source"]
    arch_metadata = output["archival_metadata"]
    if arch_metadata is None:
        arch_metadata = { "protocol": "", "protocol_date": "" }
    # TODO: type check this (at runtime, it seems to work)
    return {
        "university": {
            "Sigla": None,  # TODO: figure this out
            "Comune": context["municipality"],
            "Ubicazione": context["location"],
            "Indirizzo": context["address"],
            "Località": context["place"],
            "Data intervento": str(context["intervention_date"]),
            "Tipo di intervento": context["intervention_type"],
            "Durata": context["duration"],
            "Eseguito da": str(context["executor"]),
            "Direzione scientifica": str(context["principal_investigator"]),
            "Estensione": process_extensions(context["extension"]),
            "Numero di saggi": details["sample_number"],
            "Profondità massima": details["max_depth"],
            "Geologico": details["geology"],
            "OGD": details["historical_information_class"],
            "OGM": doc_build_data["document_source_type"],
            "Profondità falda": details["groundwater_depth"],
        },
        "building": {
            "Istituzione": coalesce_str(doc_build_data["institution"]),
            "Funzionario competente": "".join(
                [str(name) for name in context["on_site_qualified_official"]]
            )
            if context["on_site_qualified_official"] is not None
            else "",
            "Tipo di documento": doc_build_data["document_type"],
            "Protocollo": arch_metadata["protocol"],
            "Data Protocollo": arch_metadata["protocol_date"],
        },
        "scheda-intervento": {
            "id": 0  # TODO: edit this with the correct id, if exists
        },
    }

# def toMagohData(output: ArchaeologicalInterventionData) -> MagohData:
#     return {
#         # TODO: figure it out
#         "sigla": "",
#         "comune": output.municipality,
#         "ubicazione": output.location,
#         "indirizzo": output.address if output.address is not None else "",
#         "località": output.place,
#         "data intervento": format_moment_italian(output.intervention_date),
#         "tipo di intervento": output.intervention_type,
#         "durata": f"{str(output.duration)} days" if output.duration is not None else "",
#         "eseguito da": output.executor if isinstance(output.executor, str) else toMappaNaming(output.executor),
#         "direzione scientifica": toMappaNaming(output.principal_investigator),
#         "estensione": ", ".join(output.extension) if output.extension is not None else "",
#         "numero di saggi": str(output.sample_number),
#         "profondità massima": f"-{str(abs(output.max_depth))}",
#         "geologico": "" if output.geology is None else ("Sì" if output.geology else "No"),
#         "Oggetti da Disegno OGD": output.purpose_of_ogd_draw,
#         "Oggetti da museo OGM": output.purpose_of_ogm_museum,
#         "profondità di falda": "" if output.groundwater_depth is None else f"-{str(abs(output.groundwater_depth))}",
#         "Istituzione": output.institution if output.institution is not None else "",
#         "Funzionario competente": ", ".join(map(toMappaNaming, output.on_site_qualified_official)),
#         "Tipo di documento": output.document_type,
#         "protocollo": output.protocol if output.protocol is not None else "",
#         "data protocollo": output.protocol_date if output.protocol_date is not None else "",
#     }
# 
#     # additional fields :
#     #    redattore: my incredible AI
#     #    motivation: the project
#     #    
#     # reamining fields
#     #
#     # esecutore: better to leave it empty if there is a doubt 
#     # one name or a institution name
#     #
#     # all eras
#     #
#     # Fonte informazione: most of the time, has to be inferred with computation
