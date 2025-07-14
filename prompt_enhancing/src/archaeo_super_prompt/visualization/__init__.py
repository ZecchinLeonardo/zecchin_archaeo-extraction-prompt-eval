from dash import Dash, html
from pandera.typing.pandas import DataFrame

from ..types.results import ResultSchema

from .display_fields import display_results
from .file_explorer import add_file_explorer

_app = Dash()
_is_display_server_running = False


def init_complete_vizualisation_engine(results: DataFrame[ResultSchema]):
    resultChildren, c1 = display_results(results)
    fileExplorer, c2 = add_file_explorer()
    _app.layout = [
        html.Div(className="row", children=resultChildren),
        html.Div(className="row", children=fileExplorer),
    ]
    c1(_app)
    c2(_app)


def run_display_server():
    global _app
    global _is_display_server_running

    if not _is_display_server_running:
        _app.run()
