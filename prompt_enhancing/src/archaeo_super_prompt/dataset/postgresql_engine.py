import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from ..env import getenv_or_throw


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


__engine = _create_engine_from_credentials()


def get_engine():
    global __engine
    return __engine


def _import_sql(sql_path: Path):
    with sql_path.open("r") as sql_file:
        return sql_file.read()


__sampling_request = _import_sql(Path("./sql/sampling.sql"))
__get_sample_findings_request = _import_sql(Path("./sql/sample_findings.sql")).replace(
    "-- sampling-placeholder", __sampling_request
)


def get_entries(max_number: int, seed: int):
    deterministic_params = {"seed": seed, "max_number": max_number}
    intervention_data = pd.read_sql(
        __sampling_request, __engine, params=deterministic_params
    )
    findings = pd.read_sql(
        __get_sample_findings_request, __engine, params=deterministic_params
    )
    return intervention_data, findings
