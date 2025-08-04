"""Transform functions."""

import re

from .utils import (
    Date,
    InterventionDataForDateNormalizationRowSchema,
    Precision,
)


def get_day_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    pattern = (
        r"(\d{1,2})\s+([a-zA-Z]+)\s*-\s*(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})"
    )
    pattern_without_year = (
        r"(\d{1,2})\s+([a-zA-Z]+)\s*-\s*(\d{1,2})\s+([a-zA-Z]+)"
    )
    m = re.fullmatch(pattern, s)
    if m:
        day1, month1, day2, month2, year = m.groups()
        return Date(
            f"{day1}/{month1}/{year}", f"{day2}/{month2}/{year}", "day"
        )
    m = re.fullmatch(pattern_without_year, s)
    if m:
        day1, month1, day2, month2 = m.groups()
        year = row.anno
        return Date(
            f"{day1}/{month1}/{year}", f"{day2}/{month2}/{year}", "day"
        )
    return None


def get_single_day_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    pattern_without_year = r"(\d{1,2})\s+([a-zA-Z]+)"
    pattern_with_year = pattern_without_year + r"\s+(\d{4})"
    m_with_year = re.fullmatch(pattern_with_year, s)
    if m_with_year:
        day1, month1, year = m_with_year.groups()
        return Date(
            f"{day1}/{month1}/{year}", f"{day1}/{month1}/{year}", "day"
        )
    m_without_year = re.fullmatch(pattern_without_year, s)
    if m_without_year:
        day1, month1 = m_without_year.groups()
        year1 = row.anno
        return Date(
            f"{day1}/{month1}/{year1}", f"{day1}/{month1}/{year1}", "day"
        )
    return None


def get_month_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    pattern = r"([a-zA-Z]+)\s+(\d{4})\s*-\s*([a-zA-Z]+)\s+(\d{4})"
    pattern_with_year_on_right = r"([a-zA-Z]+)\s*-\s*([a-zA-Z]+)\s+(\d{4})"
    pattern_without_year = r"([a-zA-Z]+)\s*-\s*([a-zA-Z]+)"
    m = re.fullmatch(pattern, s)
    if m:
        month1, year1, month2, year2 = m.groups()
        return Date(f"{1}/{month1}/{year1}", f"{28}/{month2}/{year2}", "month")
    m = re.fullmatch(pattern_with_year_on_right, s)
    if m:
        month1, month2, year = m.groups()
        return Date(f"{1}/{month1}/{year}", f"{28}/{month2}/{year}", "month")
    m = re.fullmatch(pattern_without_year, s)
    if m:
        month1, month2 = m.groups()
        year = row.anno
        return Date(f"{1}/{month1}/{year}", f"{28}/{month2}/{year}", "month")
    return None


def get_single_month_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    pattern_without_year = r"([a-zA-Z]+)"
    pattern_with_year = pattern_without_year + r"\s+(\d{4})"
    m_with_year = re.fullmatch(pattern_with_year, s)
    if m_with_year:
        month1, year1 = m_with_year.groups()
        return Date(f"{1}/{month1}/{year1}", f"{28}/{month1}/{year1}", "month")
    m_without_year = re.fullmatch(pattern_without_year, s)
    if m_without_year:
        (month1,) = m_without_year.groups()
        year1 = row.anno
        return Date(f"{1}/{month1}/{year1}", f"{28}/{month1}/{year1}", "month")
    return None


def start_year(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    pattern = r"(\d{4})\s*-\s*"
    m = re.fullmatch(pattern, s)
    if not m:
        return None
    start_year = m.groups()
    final_year = row.anno
    return Date(f"{1}/{1}/{start_year}", f"{31}/{12}/{final_year}", "year")


def _get_d_y_m(ds: str) -> tuple[str | None, str | None, str | None]:
    d_m_y__pattern = r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})"
    d_m__pattern = r"(\d{1,2})\s+([a-zA-Z]+)"
    m_y__pattern = r"([a-zA-Z]+)\s+(\d{4})"
    y__pattern = r"(\d{4})"
    m__pattern = r"([a-zA-Z]+)"
    d__pattern = r"(\d{1,2})"
    m = re.fullmatch(d_m_y__pattern, ds)
    if m:
        day, month, year = m.groups()
        return (day, month, year)
    m = re.fullmatch(d_m__pattern, ds)
    if m:
        day, month = m.groups()
        return (day, month, None)
    m = re.fullmatch(m_y__pattern, ds)
    if m:
        month, year = m.groups()
        return (None, month, year)
    m = re.fullmatch(y__pattern, ds)
    if m:
        (year,) = m.groups()
        return None, None, year
    m = re.fullmatch(m__pattern, ds)
    if m:
        (month,) = m.groups()
        return None, month, None
    m = re.fullmatch(d__pattern, ds)
    if m:
        (day,) = m.groups()
        return day, None, None
    return None, None, None


def generic_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    split_pattern = r"\s*-\s*"
    splits = re.split(split_pattern, s)
    if len(splits) != 2:
        return None

    start, end = tuple(map(_get_d_y_m, splits))
    if start == (None, None, None) and end == (None, None, None):
        return None
    precision: Precision = "day"
    if start[0] is None and end[0] is None:
        if start[1] is None and end[1] is None:
            precision = "year"
        else:
            precision = "month"
    return Date(
        "/".join(
            (
                start[0] if start[0] is not None else str(1),
                start[1]
                if start[1] is not None
                else (end[1] if end[1] is not None else str(1)),
                start[2] if start[2] is not None else str(row.anno),
            )
        ),
        "/".join(
            (
                end[0]
                if end[0] is not None
                else (str(28) if precision == "month" else str(31)),
                end[1] if end[1] is not None else str(12),
                end[2] if end[2] is not None else str(row.anno),
            )
        ),
        precision,
    )


def generic_single_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    """Process a single period with the day, month or year precision."""
    d, m, y = _get_d_y_m(row.data_intervento.strip())
    if (d, m, y) == (None, None, None):
        return None
    y = y if y is not None else str(row.anno)
    if m is None:
        return Date(f"{1}/{1}/{y}", f"{31}/{12}/{y}", "year")
    if d is None:
        return Date(f"{1}/{m}/{y}", f"{28}/{m}/{y}", "month")
    return Date(f"{d}/{m}/{y}", f"{d}/{m}/{y}", "day")


def precised_numeric_start_date(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    """Process a precise date in the format day/month/year."""
    date_pattern = r"(\d{1,2})\/(\d{1,2})\/(\d{4})"
    m = re.fullmatch(date_pattern, row.data_intervento.strip())
    if m:
        d, m, y = m.groups()
        return Date(f"{d}/{m}/{y}", f"{d}/{m}/{y}", "day")
    return None


def before_day_month(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    """Return only the most recent day before which the intervention could happen."""
    s = row.data_intervento
    pattern = r"(?:pre|[a,A]nte|prima di|prima del)\s+(.*)"
    m = re.fullmatch(pattern, s)
    if not m:
        return None
    (final_date_str,) = m.groups()
    d, m, y = _get_d_y_m(final_date_str.rstrip())
    y = y if y is not None else row.anno
    precision: Precision = (
        "day" if d is not None else ("month" if m is not None else "year")
    )
    m = m if m is not None else 12
    d = d if d is not None else (31 if precision == "year" else 28)
    return Date(
        "<UNKNOWN>",
        f"{d}/{m}/{y}",
        "day",
    )
