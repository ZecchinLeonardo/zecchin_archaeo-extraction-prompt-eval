from pathlib import Path
from dash import Dash, html, callback, Output, Input, dcc
import flask

from .types import DashComponent

from ..cache import get_cache_dir_for

_STATIC_DIR = get_cache_dir_for("external", "pdfs").resolve()


def _get_file_endpoints():
    return {
        dirp.name: [str((Path("/") / p.parent.name) / p.name) for p in dirp.iterdir()]
        for dirp in _STATIC_DIR.iterdir()
    }


def add_file_explorer() -> DashComponent:
    files_dirs = _get_file_endpoints()

    DD_INTERVENTION_ID, DD_FILENAME_TAG_ID, PDF_VIEWER_TAG_ID = (
        "dd-interv-id",
        "dd-filename",
        "pdf-viewer",
    )
    PDF_FILE_ENDPOINT = "pdf-file"

    new_layout = [
        html.H1("Source PDF Explorer"),
        dcc.Dropdown(list(files_dirs.keys()), id=DD_INTERVENTION_ID),
        dcc.Dropdown(id=DD_FILENAME_TAG_ID),
        html.Iframe(
            style={"width": "100%", "height": "600px", "border": "none"},
            id=PDF_VIEWER_TAG_ID,
        ),
    ]

    def init_callbacks(app: Dash):
        @callback(
            Output(DD_FILENAME_TAG_ID, "options"),
            Input(DD_INTERVENTION_ID, "value"),
        )
        def updatePdfList(intervention_id: str | None) -> list[str]:
            if intervention_id is None:
                return []
            return [str(p) for p in files_dirs[intervention_id]]

        @app.server.route(f"/{PDF_FILE_ENDPOINT}/<path:filename>")
        def serve_pdf_file(filename):
            return flask.send_from_directory(str(_STATIC_DIR), filename)

        @callback(
            Output(PDF_VIEWER_TAG_ID, "src"),
            Input(DD_FILENAME_TAG_ID, "value"),
        )
        def updatePdfSrc(path: str | None) -> str | None:
            if path is None:
                return None
            return "/" + PDF_FILE_ENDPOINT + path

        @callback(
            Output(DD_FILENAME_TAG_ID, "value"),
            Input(DD_INTERVENTION_ID, "value"),
        )
        def updateDefaultFilename(intervention_id: str) -> str | None:
            options = updatePdfList(intervention_id)
            return options[0] if options else None

        # remove unused warning as they are used within the callback decorator
        serve_pdf_file = serve_pdf_file
        updatePdfSrc = updatePdfSrc
        updateDefaultFilename = updateDefaultFilename

    return new_layout, init_callbacks
