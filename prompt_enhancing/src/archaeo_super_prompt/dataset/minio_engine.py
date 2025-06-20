from pathlib import Path
from typing import List
from minio import Minio
from ..env import getenv_or_throw

_host = getenv_or_throw("MINIO_HOST")
_user = getenv_or_throw("MINIO_ROOT_USER")
_password = getenv_or_throw("MINIO_ROOT_PASSWORD")

__client = Minio(
    _host,
    access_key=_user,
    secret_key=_password,
    secure=not _host.startswith("localhost"),
)

BUCKET_NAME = "training-reports"
if not __client.bucket_exists(BUCKET_NAME):
    __client.make_bucket(BUCKET_NAME)


def download_files(intervention_id: int) -> List[Path]:
    pdf_store_dir = Path("./.cache/pdfs/")
    if not pdf_store_dir.exists():
        pdf_store_dir.mkdir(parents=True, exist_ok=True)

    dirpath = Path("./" + str(intervention_id))

    output_pathdir = pdf_store_dir / dirpath
    if output_pathdir.exists():
        return [f for f in output_pathdir.iterdir()]

    files = __client.list_objects(BUCKET_NAME, prefix=str(dirpath),
                                  recursive=True)

    def download_and_return():
        for file in files:
            object_name = file.object_name
            if object_name is None:
                continue

            output_path = pdf_store_dir / object_name
            _ = (__client.fget_object(BUCKET_NAME, object_name, str(output_path)),)
            yield output_path

    return [p for p in download_and_return()]
