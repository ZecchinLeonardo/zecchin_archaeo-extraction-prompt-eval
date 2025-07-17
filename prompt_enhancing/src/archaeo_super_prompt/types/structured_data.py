"""Types related to the wanted structured data in the dataset."""

from collections.abc import Iterator
from typing import Any, NamedTuple, cast
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series
import pandas as pd


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
        "university.Numero di saggi": pa.Column(
            "UInt32", pa.Check.ge(0), nullable=True
        ),
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


class OutputStructuredDataSchema(pa.DataFrameModel):
    """Schema of the intervention target metadata in the dataset."""
    
    id: Series[int]
    university__Sigla: Series[str | None]
    university__Comune: Series[str | None]
    university__Ubicazione: Series[str | None]
    university__Indirizzo: Series[str | None]
    university__Località: Series[str | None]
    university__Data_intervento: Series[str | None]
    university__Tipo_di_intervento: Series[str | None]
    university__Durata: Series[str | None]
    university__Eseguito_da: Series[str | None]
    university__Direzione_scientifica: Series[str | None]
    university__Estensione: Series[str | None]
    university__Numero_di_saggi: Series[pd.UInt32Dtype | None]
    university__Profondità_massima: Series[float | None]
    university__Geologico: Series[bool | None]
    university__OGD: Series[str | None]
    university__OGM: Series[str | None]
    university__Profondità_falda: Series[float | None]
    building__Istituzione: Series[str | None]
    building__Funzionario_competente: Series[str | None]
    building__Tipo_di_documento: Series[str | None]
    building__Protocollo: Series[str | None]
    building__Data_Protocollo: Series[str | None]


class DatasetAnswerSchema(NamedTuple):
    """Schema of a row in the answer dataframe loadable from the dataset."""

    id: int
    university__Sigla: str | None
    university__Comune: str | None
    university__Ubicazione: str | None
    university__Indirizzo: str | None
    university__Località: str | None
    university__Data_intervento: str | None
    university__Tipo_di_intervento: str | None
    university__Durata: str | None
    university__Eseguito_da: str | None
    university__Direzione_scientifica: str | None
    university__Estensione: str | None
    university__Numero_di_saggi: pd.UInt32Dtype | None
    university__Profondità_massima: float | None
    university__Geologico: bool | None
    university__OGD: str | None
    university__OGM: str | None
    university__Profondità_falda: float | None
    building__Istituzione: str | None
    building__Funzionario_competente: str | None
    building__Tipo_di_documento: str | None
    building__Protocollo: str | None
    building__Data_Protocollo: str | None


def outputStructuredDataSchema_itertuples(
    df: DataFrame[OutputStructuredDataSchema],
):
    """Type-safe wrapper of DataFrame.itertuples."""
    return cast(Iterator[DatasetAnswerSchema], df.itertuples())


ExtractedStructuredDataSeries = dict[str, Any]
