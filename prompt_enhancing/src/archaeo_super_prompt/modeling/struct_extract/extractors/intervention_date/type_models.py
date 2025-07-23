"""Types for dates with semantic information."""
import pydantic
from typing import Literal

type Mese = Literal[
    "Gennaio",
    "Febbraio",
    "Marzo",
    "Aprile",
    "Maggio",
    "Giugno",
    "Luglio",
    "Agosto",
    "Settembre",
    "Ottobre",
    "Novembre",
    "Dicembre",
]

ITALIAN_MONTHS: list[Mese] = [
    "Gennaio",
    "Febbraio",
    "Marzo",
    "Aprile",
    "Maggio",
    "Giugno",
    "Luglio",
    "Agosto",
    "Settembre",
    "Ottobre",
    "Novembre",
    "Dicembre",
]


class Data(pydantic.BaseModel):
    """Un data. A volte, il giorno o il mese possono avere un valore artificiale quando la precisione non consente di prevedere questi campi."""

    giorno: int
    mese: Mese
    anno: int

type Precision = Literal["day", "month", "year"]
type Precisione = Literal["giorno", "mese", "anno"]
