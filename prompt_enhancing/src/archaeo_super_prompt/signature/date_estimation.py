import pydantic
from typing import Literal, Union

FebruaryDay = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
BigMonthDay = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
SmallMonthDay = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

class YearDayInBigMonth(pydantic.BaseModel):
    year: int
    month: Literal[1, 3, 5, 7, 8, 10, 12]
    day: BigMonthDay

class YearDayInSmallMonth(pydantic.BaseModel):
    year: int
    month: Literal[4, 6, 9, 11]
    day: SmallMonthDay

class YearDayInFebruary(pydantic.BaseModel):
    year: int
    month: Literal[2]
    day: FebruaryDay

Month = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class YearAndMonth(pydantic.BaseModel):
    year: int
    month: Month

class Year(pydantic.BaseModel):
    year: int

Date = Union[
    Year, YearAndMonth, YearDayInBigMonth, YearDayInSmallMonth, YearDayInFebruary
]


class LatestEstimatedPastMoment(pydantic.BaseModel):
    """Estimate of the latest moment when an event could have happened. This
    can be the exact moment and it has to be precised with the precision field
    (setting it with the "During" value). The precision depends on the
    information the document provides.
    For describing the moment, the year is a mandatory information and the
    precision of the month or the precise date is optional and has to be done
    only if it can be figured out from the document.
    """
    precision: Literal["Before", "During"]
    date: Date


ITALIAN_MONTHS = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
]

def format_moment_italian(moment: LatestEstimatedPastMoment) -> str:
    prefix = "Prima del " if moment.precision == "Before" else ""

    if isinstance(moment.date, Year):
        return f"{prefix}il {moment.date.year}"
    if isinstance(moment.date, YearAndMonth):
        month_str = ITALIAN_MONTHS[moment.date.month - 1]
        return f"{prefix}{month_str} {moment.date.year}"
    else:
        month_str = ITALIAN_MONTHS[moment.date.month - 1]
        return f"{prefix}{moment.date.day} {month_str} {moment.date.year}"
