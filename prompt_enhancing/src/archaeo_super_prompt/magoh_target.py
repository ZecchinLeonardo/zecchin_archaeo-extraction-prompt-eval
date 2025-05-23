from typing import List, TypedDict

from archaeo_super_prompt.signature.date_estimation import format_moment_italian
from archaeo_super_prompt.signature.name import toMappaNaming

from .signature.arch_extract_type import ArchaeologicalInterventionData

MagohData = TypedDict(
    "MagohData",
    {
        "sigla": str, # according to Francesco, analyzing this field is skipable as in most of the case this is not written, and every document is relazione di scava
        "comune": str,
        "ubicazione": str,
        "indirizzo": str,
        "località": str,
        "data intervento": str,
        "tipo di intervento": str,
        "durata": str,
        "eseguito da": str,
        "direzione scientifica": str,
        "estensione": str,
        "numero di saggi": str,
        "profondità massima": str,
        "geologico": str,
        "Oggetti da Disegno OGD": str,
        "Oggetti da museo OGM": str,
        "profondità di falda": str,
        "Istituzione": str,
        "Funzionario competente": str,
        "Tipo di documento": str,
        "protocollo": str,
        "data protocollo": str,
    },
)


def dID_objects_processing(raw: List[str]):
    if len(raw) == 0:
        return ""
    else:
        return f"Sì ({', '.join(raw)})"


def toMagohData(output: ArchaeologicalInterventionData) -> MagohData:
    return {
        "sigla": "",  # TODO: figure it out
        "comune": output.municipality,
        "ubicazione": output.location,
        "indirizzo": output.address if output.address is not None else "",
        "località": output.place,
        "data intervento": format_moment_italian(output.intervention_date),
        "tipo di intervento": output.intervention_type,
        "durata": f"{str(output.duration)} days" if output.duration is not None else "",
        "eseguito da": output.executor if isinstance(output.executor, str) else toMappaNaming(output.executor),
        "direzione scientifica": toMappaNaming(output.principal_investigator),
        "estensione": ", ".join(output.extension),
        "numero di saggi": str(output.sample_number),
        "profondità massima": f"-{str(abs(output.max_depth))}",
        "geologico": "" if output.geology is None else ("Sì" if output.geology else "No"),
        "Oggetti da Disegno OGD": output.purpose_of_ogd_draw,
        "Oggetti da museo OGM": output.purpose_of_ogm_museum,
        "profondità di falda": "" if output.groundwater_depth is None else f"-{str(abs(output.groundwater_depth))}",
        "Istituzione": output.institution if output.institution is not None else "",
        "Funzionario competente": ", ".join(map(toMappaNaming, output.on_site_qualified_official)),
        "Tipo di documento": output.document_type,
        "protocollo": output.protocol if output.protocol is not None else "",
        "data protocollo": output.protocol_date if output.protocol_date is not None else "",
    }

    # additional fields :
    #    redattore: my incredible AI
    #    motavioon: the project
    #    
    # reamining fields
    #
    # esecutore: better to leave it empty if there is a doubt 
    # one name or a institution name
    #
    # all eras
    #
    # Fonte informazione: most of the time, has to be inferred with computation
