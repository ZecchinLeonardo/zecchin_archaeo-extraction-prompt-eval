import pydantic
from typing import Literal, Union, override

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

ITALIAN_MONTHS = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
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
    date: str
    # TODO: refine the type when the engine enable that
    # date: Date
    
    @override
    def __str__(self):
        prefix = "Prima del " if self.precision == "Before" else ""
        return f"{prefix}{self.date}"
        if isinstance(self.date, Year):
            return f"{prefix}{self.date.year}"
        if isinstance(self.date, YearAndMonth):
            month_str = ITALIAN_MONTHS[self.date.month - 1]
            return f"{prefix}{month_str} {self.date.year}"
        else:
            month_str = ITALIAN_MONTHS[self.date.month - 1]
            return f"{prefix}{self.date.day} {month_str} {self.date.year}"

