from typing import Any
import pandera.pandas as pa

def _negativeFloatColumn():
    return pa.Column(float, pa.Check.lt(0.000001), nullable=True)

structuredDataSchema = pa.DataFrameSchema(
    {
        "scheda_intervento.id": pa.Column(int),
        "university.Sigla": pa.Column(str, nullable=True),
        "university.Comune": pa.Column(str, nullable=True),
        "university.Ubicazione": pa.Column(str, nullable=True),
        "university.Indirizzo": pa.Column(str, nullable=True),
        "university.Località": pa.Column(str, nullable=True),
        "university.Data intervento": pa.Column(str, nullable=True),
        "university.Tipo di intervento": pa.Column(str, nullable=True),
        "university.Durata": pa.Column(str, nullable=True),
        "university.Eseguito da": pa.Column(str, nullable=True),
        "university.Direzione scientifica": pa.Column(str, nullable=True),
        "university.Estensione": pa.Column(str, nullable=True),
        "university.Numero di saggi": pa.Column('UInt32', pa.Check.ge(0), nullable=True),
        "university.Profondità massima": _negativeFloatColumn(),
        "university.Geologico": pa.Column("boolean", nullable=True),
        "university.OGD": pa.Column(str, nullable=True),
        "university.OGM": pa.Column(str, nullable=True),
        "university.Profondità falda": _negativeFloatColumn(),
        "check.Preistoria": pa.Column(bool),
        "check.Età Protostorica": pa.Column(bool),
        "check.Età Etrusca": pa.Column(bool),
        "check.Età Romana": pa.Column(bool),
        "check.Età Tardoantica": pa.Column(bool),
        "check.Alto Medioevo": pa.Column(bool),
        "check.Basso Medioevo": pa.Column(bool),
        "check.Età Moderna": pa.Column(bool),
        "check.Età Contemporanea": pa.Column(bool),
        "check.Non identificati": pa.Column(bool),
        "building.Istituzione": pa.Column(str, nullable=True),
        "building.Funzionario competente": pa.Column(str, nullable=True),
        "building.Tipo di documento": pa.Column(str, nullable=True),
        "building.Protocollo": pa.Column(str, nullable=True),
        "building.Data Protocollo": pa.Column(str, nullable=True),
    }
)

outputStructuredDataSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(int),
        "university.Sigla": pa.Column(str, nullable=True),
        "university.Comune": pa.Column(str, nullable=True),
        "university.Ubicazione": pa.Column(str, nullable=True),
        "university.Indirizzo": pa.Column(str, nullable=True),
        "university.Località": pa.Column(str, nullable=True),
        "university.Data intervento": pa.Column(str, nullable=True),
        "university.Tipo di intervento": pa.Column(str, nullable=True),
        "university.Durata": pa.Column(str, nullable=True),
        "university.Eseguito da": pa.Column(str, nullable=True),
        "university.Direzione scientifica": pa.Column(str, nullable=True),
        "university.Estensione": pa.Column(str, nullable=True),
        "university.Numero di saggi": pa.Column('UInt32', pa.Check.ge(0), nullable=True),
        "university.Profondità massima": _negativeFloatColumn(),
        "university.Geologico": pa.Column("boolean", nullable=True),
        "university.OGD": pa.Column(str, nullable=True),
        "university.OGM": pa.Column(str, nullable=True),
        "university.Profondità falda": _negativeFloatColumn(),
        "building.Istituzione": pa.Column(str, nullable=True),
        "building.Funzionario competente": pa.Column(str, nullable=True),
        "building.Tipo di documento": pa.Column(str, nullable=True),
        "building.Protocollo": pa.Column(str, nullable=True),
        "building.Data Protocollo": pa.Column(str, nullable=True),
    }
)

ExtractedStructuredDataSeries = dict[str, Any]
