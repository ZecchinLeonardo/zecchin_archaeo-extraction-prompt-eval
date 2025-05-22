from typing import List, TypedDict

from archaeo_super_prompt.signature import ExtractionArcheoData

MagohData = TypedDict(
    "MagohData",
    {
        "scheda_intervento": str, # do not output it
        "sigla": str, # from Francesco, this is not cool as in most of the case
        # , this is not written, and every document is relazione di scava 
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


def toMagohData(output: ExtractionArcheoData) -> MagohData:
    return {
        "scheda_intervento": "",  # TODO: fill with filenmae
        "sigla": "",  # TODO: figure it out
        "comune": output.municipality,
        "ubicazione": output.location,
        "indirizzo": output.address if output.address is not None else "",
        "località": output.place,
        "data intervento": ", ".join(output.intervention_date),
        "tipo di intervento": output.intervention_type,
        "durata": f"{str(output.duration)} days" if output.duration is not None else "",
        "eseguito da": output.done_since,
        "direzione scientifica": output.scientific_direction,
        "estensione": ", ".join(output.extension),
        "numero di saggi": str(output.test_number),
        "profondità massima": str(output.max_depth),
        "geologico": output.geology,
        "Oggetti da Disegno OGD": dID_objects_processing(output.diD_stuff),
        "Oggetti da museo OGM": ", ".join(output.ogm_museum_stuff),
        "profondità di falda": output.falda_depth
        if output.falda_depth is not None
        else "",
        "Istituzione": output.institution if output.institution is not None else "",
        "Funzionario competente": ", ".join(output.on_site_qualified_official),
        "Tipo di documento": output.document_type,
        "protocollo": output.protocol if output.protocol is not None else "",
        "data protocollo": output.protocol_date
        if output.protocol_date is not None
        else "",
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
