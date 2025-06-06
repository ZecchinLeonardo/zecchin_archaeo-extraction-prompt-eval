from .target_types import MagohData
from .models.main_pipeline import ExtractedInterventionData

def toMagohData(output: ExtractedInterventionData) -> MagohData:
    return {
        "university": output['university'],
        "building": output['build'],
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
