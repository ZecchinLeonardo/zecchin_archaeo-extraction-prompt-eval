"""Transforming functions to get a normalized duration of an excavation."""

from collections.abc import Callable

from archaeo_super_prompt.dataset.normalization.intervention_date.utils import (
    Duration,
)
import re

giorno_pattern = re.compile(r"(\d+)\s*(?:giorn[oi]|gg|g)")
meso_pattern = re.compile(r"(\d+)\s*mes[ei]")
year_pattern = re.compile(r"(\d+)\s*ann[oi]")
settimane_pattern = re.compile(r"(\d+)\s*settiman[ae]")


def _pipe_matches(
    callbacks: list[tuple[re.Pattern[str], Callable[[int], Duration]]],
    string: str,
):
    for p, callback in callbacks:
        m = p.fullmatch(string)
        if m:
            (number,) = m.groups()
            return callback(int(number))
    return string


def parse_duration(duration: int | str | None) -> Duration | str | None:
    """Return a numeric duration, if possible.

    If the value is null-like, then return None, if the value can be parsed,
    then return a Duration object. Else, return the unparsed string.
    """
    if duration is None:
        return None
    if isinstance(duration, int):
        return Duration(int(duration), "day")
    duration = duration.strip()
    if duration == "":
        return None
    if duration.isdigit():
        return Duration(int(duration), "day")
    return _pipe_matches(
        [
            (giorno_pattern, lambda days: Duration(days, "day")),
            (meso_pattern, lambda months: Duration(months, "month")),
            (year_pattern, lambda years: Duration(years, "year")),
            (settimane_pattern, lambda weeks: Duration(7 * weeks, "day")),
        ],
        duration,
    )
