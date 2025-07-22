"""Structured data loading from a remote dataset.

This module manages the interaction with the postgresql database to load a
pandas DataFrame. The sqlalchemy library with the psycopg2 engine are used.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import Engine, create_engine

from ..config.env import getenv_or_throw


def _create_engine_from_credentials():
    DIALECT = "postgresql"
    DRIVER = "psycopg2"
    writing_db_user = getenv_or_throw("PG_SUPERUSER")
    db_name = getenv_or_throw("PG_DB_NAME")
    db_user_password = getenv_or_throw("PG_DB_PASSWORD")

    db_host = getenv_or_throw("PG_DB_HOST")
    db_port = getenv_or_throw("PG_DB_PORT")

    return create_engine(
        f"{DIALECT}+{DRIVER}://{writing_db_user}:{db_user_password}@{db_host}:{db_port}/{db_name}"
    )


__engine: Engine | None = None


def _get_engine():
    global __engine
    if __engine is None:
        __engine = _create_engine_from_credentials()
    return __engine


def _import_sql(sql_path: Path):
    with sql_path.open("r") as sql_file:
        return sql_file.read()


__module_dir = Path(__file__).parent

__seed_setting_request = _import_sql(__module_dir / Path("sql/setseed.sql"))
__sampling_request = _import_sql(__module_dir / Path("sql/sampling.sql"))
__sampling_on_recents_request = _import_sql(
    __module_dir / Path("sql/sampling_on_recents.sql")
)
__get_sample_findings_request = _import_sql(
    __module_dir / Path("sql/sample_findings.sql")
)
__get_intervention_with_ids = _import_sql(
    __module_dir / Path("sql/select_ids.sql")
)
__get_findings_with_ids = _import_sql(
    __module_dir / Path("./sql/select_findings_ids.sql")
)


def get_entries(max_number: int, seed: float, only_recent_entries=False):
    """Fetch from the remote database a set of samples of interventions."""
    engine = _get_engine()
    findings_request = __get_sample_findings_request.replace(
        "-- sampling-placeholder",
        __sampling_request
        if not only_recent_entries
        else __sampling_on_recents_request,
    )
    deterministic_params = {"seed": seed, "max_number": max_number}
    print("Fetching structured intervention data...")
    intervention_data = pd.read_sql(
        __seed_setting_request + "\n" + __sampling_request,
        engine,
        params=deterministic_params,
    )
    print("Fetching done!")
    print("Fetching saved findings for each intervention...")
    findings = pd.read_sql(
        __seed_setting_request + "\n" + findings_request,
        engine,
        params=deterministic_params,
    )
    print("Fetching done!")
    return intervention_data, findings


def get_entries_with_ids(ids: set[int]):
    """Fetch on the db the metadata of the intervention with the given ids."""
    engine = _get_engine()
    id_set_for_request = tuple(ids)
    interventions = pd.read_sql(
        __get_intervention_with_ids,
        engine,
        params={"intervention_ids": id_set_for_request},
    )
    findings = pd.read_sql(
        __get_findings_with_ids,
        engine,
        params={"intervention_ids": id_set_for_request}
    )
    return interventions, findings
