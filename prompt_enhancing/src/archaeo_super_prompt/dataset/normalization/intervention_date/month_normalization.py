"""Code to transform a noisy month string into a numerical month."""

from thefuzz import process

ITALIAN_MONTHS = list(
    map(
        lambda m: m.lower(),
        [
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
        ],
    )
)


def to_int_month(month_str: str) -> int:
    if month_str.isdigit():
        return int(month_str)
    norm = month_str.lower().strip()
    if norm in ITALIAN_MONTHS:
        return ITALIAN_MONTHS.index(norm) + 1
    best_month_list: list[tuple[str, int]] = process.extractBests(
        norm, ITALIAN_MONTHS, limit=1, score_cutoff=90
    )
    if len(best_month_list) == 0:
        raise Exception(f"Cannot parse this month: '{month_str}'")
    return ITALIAN_MONTHS.index(best_month_list[0][0]) + 1
