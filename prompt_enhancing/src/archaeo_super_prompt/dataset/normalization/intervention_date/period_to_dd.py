"""Fix functions for transforming a period of intervention into a start date and a duration."""

from .utils import Duration
import pandas as pd
from datetime import timedelta


def fix_start_and_duration(df: pd.DataFrame):
    """Fix the intervention date and duration when it is sure that the whole intervention period has been written."""
    df = df[~(df["end_date"] - df["start_date"] < timedelta(0))]
    delta = df["end_date"] - df["start_date"]
    df = df.assign(
        norm_duration=delta.apply(
            lambda d: Duration(d.days + 1, "day")
            if isinstance(d, timedelta)
            else None
        ),
    ).where(
        (delta > timedelta(0))
        & (df["precision"] == "day")
        & (df["norm_duration"].isnull()),
        df,
    )
    return df.assign(
        end_date=df["start_date"],
    ).where((delta > timedelta(0)) & (df["precision"] == "day"), df)
