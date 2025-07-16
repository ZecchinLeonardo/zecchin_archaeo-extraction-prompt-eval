from typing import cast
from minio import Minio
from pathlib import Path
import re

from ..utils.cache import get_cache_dir_for
from ..config.env import getenv_or_throw

_host = getenv_or_throw("MINIO_HOST")
_user = getenv_or_throw("MINIO_ROOT_USER")
_password = getenv_or_throw("MINIO_ROOT_PASSWORD")

__client = None

BUCKET_NAME = "training-reports"


def _init_client():
    c = Minio(
        _host,
        access_key=_user,
        secret_key=_password,
        secure=not _host.startswith("localhost"),
    )

    if not c.bucket_exists(BUCKET_NAME):
        c.make_bucket(BUCKET_NAME)
    return c


# Allow only letters, digits, underscores, hyphens, and dots
SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")  # MATCHES UNSAFE chars


def sanitize_filename(filename):
    return SAFE_FILENAME_PATTERN.sub(
        "_", filename
    )  # Replace unsafe chars with underscore


def download_files(intervention_id: int) -> list[Path]:
    global __client

    pdf_store_dir = get_cache_dir_for("external", "pdfs")
    if not pdf_store_dir.exists():
        pdf_store_dir.mkdir(parents=True, exist_ok=True)

    dirpath = Path("./" + str(intervention_id))

    output_pathdir = pdf_store_dir / dirpath
    if output_pathdir.exists():
        files = [f for f in output_pathdir.iterdir()]
        if files:
            return files

    if __client is None:
        __client = _init_client()
    files = __client.list_objects(
        BUCKET_NAME, prefix=str(dirpath), recursive=True
    )

    def download_and_return():
        for file in files:
            object_name = file.object_name
            if object_name is None:
                continue
            it_id_subdir, filename = object_name.split("/")
            output_path = (pdf_store_dir / it_id_subdir) / sanitize_filename(
                filename
            )
            _ = (
                cast(Minio, __client).fget_object(
                    BUCKET_NAME, object_name, str(output_path)
                ),
            )
            yield output_path

    return [p for p in download_and_return()]
