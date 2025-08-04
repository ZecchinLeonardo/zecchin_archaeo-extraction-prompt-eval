"""Transform functions."""

import re

from .utils import Date, InterventionDataForDateNormalizationRowSchema


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


def generic_period(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    split_pattern = r"\s*-\s*"
    splits = re.split(split_pattern, s)
    if len(splits) != 2:
        return None
    d_m_y__pattern = r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})"
    d_m__pattern = r"(\d{1,2})\s+([a-zA-Z]+)"
    m_y__pattern = r"([a-zA-Z]+)\s+(\d{4})"
    m__pattern = r"([a-zA-Z]+)"
    d__pattern = r"(\d{1,2})"

    def get_d_y_m(ds: str) -> tuple[str | None, str | None, str | None]:
        m = re.fullmatch(d_m_y__pattern, ds)
        if m:
            return (m[0], m[1], m[2])
        m = re.fullmatch(d_m__pattern, ds)
        if m:
            return (m[0], m[1], None)
        m = re.fullmatch(m_y__pattern, ds)
        if m:
            return (None, m[0], m[1])
        m = re.fullmatch(m__pattern, ds)
        if m:
            return None, m[0], None
        m = re.fullmatch(d__pattern, ds)
        if m:
            return m[0], None, None
        return None, None, None

    start, end = tuple(map(get_d_y_m, splits))
    if start == (None, None, None) and end == (None, None, None):
        return None
    precision = "day"
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
                end[0] if end[0] is not None else str(28),
                end[1] if end[1] is not None else str(12),
                end[2] if end[2] is not None else str(row.anno),
            )
        ),
        precision,
    )


def before_day_month(
    row: InterventionDataForDateNormalizationRowSchema,
) -> Date | None:
    s = row.data_intervento
    pattern = r"pre\s+(\d{1,2})\s+([a-zA-Z]+)"
    m = re.fullmatch(pattern, s)
    if not m:
        return None
    final_day, final_month = m.groups()
    year = row.anno
    # TODO: compute it with datetime
    return Date(
        f"{1}/{final_month}-1/{year}",
        f"{final_day}/{final_month}/{year}",
        "day",
    )
