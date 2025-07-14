from typing import Literal
from pydantic import BaseModel


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
    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


class CompleteEntity(BaseModel):
    entity: NerXXLEntities
    word: str
    start: int
    end: int
