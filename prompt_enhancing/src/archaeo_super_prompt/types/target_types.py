from typing import TypedDict

MagohUniversityData = TypedDict(
    "MagohUniversityData",
    {
        "Sigla": str | None,
        "Comune": str,
        "Ubicazione": str,
        "Indirizzo": str | None,
        "Località": str | None,
        "Data intervento": str,
        "Tipo di intervento": str,
        "Durata": str | None,
        "Eseguito da": str | None,
        "Direzione scientifica": str | None,
        "Estensione": str | None,
        "Numero di saggi": int,  # unsigned
        "Profondità massima": float | None,  # absolute value but negative
        "Geologico": bool | None,
        "OGD": str,
        "OGM": str,
        "Profondità falda": float | None,
    },
)

# TODO: MagohCheckedEras

MagohDocumentBuildingData = TypedDict(
    "MagohDocumentBuildingData",
    {
        "Istituzione": str,
        "Funzionario competente": str,
        "Tipo di documento": str,
        "Protocollo": str,
        "Data Protocollo": str,
    },
)

MagohFindingScheme = TypedDict(
    "MagohFindingScheme",
    {
        "I Livello": str | None,
        "II Livello": str | None,
        "III Livello": str | None,
        "Datazione": int | None,
        "Datazione Finale": int | None,
    },
)

class MagohArtificialRecordData(TypedDict):
    id: int

# TODO: add findings to MagohData
class MagohData(TypedDict):
    university: MagohUniversityData
    building: MagohDocumentBuildingData
    scheda_intervento: MagohArtificialRecordData
