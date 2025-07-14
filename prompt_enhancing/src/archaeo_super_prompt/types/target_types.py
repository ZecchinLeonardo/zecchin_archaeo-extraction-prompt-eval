from typing import Optional, TypedDict

MagohUniversityData = TypedDict(
    "MagohUniversityData",
    {
        "Sigla": Optional[str],
        "Comune": str,
        "Ubicazione": str,
        "Indirizzo": Optional[str],
        "Località": Optional[str],
        "Data intervento": str,
        "Tipo di intervento": str,
        "Durata": Optional[str],
        "Eseguito da": Optional[str],
        "Direzione scientifica": Optional[str],
        "Estensione": Optional[str],
        "Numero di saggi": int,  # unsigned
        "Profondità massima": Optional[float],  # absolute value but negative
        "Geologico": Optional[bool],
        "OGD": str,
        "OGM": str,
        "Profondità falda": Optional[float],
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
        "I Livello": Optional[str],
        "II Livello": Optional[str],
        "III Livello": Optional[str],
        "Datazione": Optional[int],
        "Datazione Finale": Optional[int],
    },
)

MagohArtificialRecordData = TypedDict("MagohArtificialRecordData", {"id": int})

# TODO: add findings to MagohData
MagohData = TypedDict(
    "MagohData",
    {
        "university": MagohUniversityData,
        "building": MagohDocumentBuildingData,
        "scheda_intervento": MagohArtificialRecordData,
    },
)
