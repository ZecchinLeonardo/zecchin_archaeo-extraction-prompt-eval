"""Custom types related to NER models."""

from typing import Literal, NamedTuple
from pydantic import BaseModel
from pandera.pandas import DataFrameModel

from ...types.pdfchunks import PDFChunkDatasetSchema
from ...types.thesaurus import ThesaurusProvider


NerXXLEntities = Literal[
    "INDIRIZZO",
    "VALUTA",
    "CVV",
    "NUMERO_CONTO",
    "BIC",
    "IBAN",
    "STATO",
    "NOME",
    "COGNOME",
    "CODICE_POSTALE",
    "IP",
    "ORARIO",
    "URL",
    "LUOGO",
    "IMPORTO",
    "EMAIL",
    "PASSWORD",
    "NUMERO_CARTA",
    "TARGA_VEICOLO",
    "DATA_NASCITA",
    "DATA_MORTE",
    "RAGIONE_SOCIALE",
    "ETA",
    "DATA",
    "PROFESSIONE",
    "PIN",
    "NUMERO_TELEFONO",
    "FOGLIO",
    "PARTICELLA",
    "CARTELLA_CLINICA",
    "MALATTIA",
    "MEDICINA",
    "CODICE_FISCALE",
    "NUMERO_DOCUMENTO",
    "STORIA_CLINICA",
    "AVV_NOTAIO",
    "P_IVA",
    "LEGGE",
    "TASSO_MUTUO",
    "N_SENTENZA",
    "MAPPALE",
    "SUBALTERNO",
    "REGIME_PATRIMONIALE",
    "STATO_CIVILE",
    "BANCA",
    "BRAND",
    "NUM_ASSEGNO_BANCARIO",
    "IMEI",
    "N_LICENZA",
    "IPV6_1",
    "MAC",
    "USER_AGENT",
    "TRIBUNALE",
    "STRENGTH",
    "FREQUENZA",
    "DURATION",
    "DOSAGGIO",
    "FORM",
]


class NerOutput(BaseModel):
    """The output of the NER model, which is often only a chunk of a complete entity."""

    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


class CompleteEntity(BaseModel):
    """One entity containing with all its chunks that are merged."""

    entity: NerXXLEntities
    word: str
    start: int
    end: int


class EntitiesPerChunkSchema(DataFrameModel):
    """Each row is related to a chunk and contains its identified named entities."""

    named_entities: list[CompleteEntity]


class ChunksWithEntities(PDFChunkDatasetSchema, EntitiesPerChunkSchema):
    """The union of the two dataframes."""

    pass


class ChunksWithThesaurus(PDFChunkDatasetSchema):
    """For each filtered chunk, a list of the identified thesaurus.

    The list can be empty if no thesaurus has been identified in the chunk but
    named entities in the type group of interest have been identified. This
    enable to keep chunks to be read by the LLM if no fuzzymatched thesaurus
    has been identified.

    The list represents a set and contains the identifiers of the thesaurus.
    """

    identified_thesaurus: list[int]


class NamedEntityField(NamedTuple):
    """Data for a structured data field with terms identifiable by NER.

    Thesaurus values is a frozen function which give the list of thesaurus with
    their related identifier.
    """

    name: str
    compatible_entities: set[NerXXLEntities]
    thesaurus_values: ThesaurusProvider
