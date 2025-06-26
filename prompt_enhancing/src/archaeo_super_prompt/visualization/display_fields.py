from dash import Dash, html, callback, Output, Input, dash_table, dcc
import plotly.express as px
from pandera.typing.pandas import DataFrame

from archaeo_super_prompt.types.results import ResultSchema
from .prettify_field_names import prettify_field_names

_app = Dash()
_is_display_server_running = False


def display_results(score_results: DataFrame[ResultSchema]):
    global _app

    score_results = prettify_field_names(score_results)

    field_grouping_keys = ["field_name", "evaluation_method"]

    resultsPerField = {
        fieldName: {
            "method": evalMethod,
            "table": resultForField.drop(columns=field_grouping_keys),
        }
        for (fieldName, evalMethod), resultForField in score_results.groupby(
            field_grouping_keys
        )
    }
    fieldNames = list(resultsPerField.keys())

    DEFAULT_SELECTED_FIELD = "Comune"

    _app.layout = [
        html.H1(children="Results", style={"textAlign": "center"}),
        html.H2(children="Global results"),
        dcc.Graph(
            figure=px.histogram(
                score_results, y="field_name", x="metric_value", histfunc="avg"
            )
        ),
        html.H2(children="Per field results"),
        dcc.Dropdown(fieldNames, DEFAULT_SELECTED_FIELD, id="dropdown-selection"),
        html.H3(children="Evaluation method used"),
        html.Blockquote(id="eval-method-description"),
        dash_table.DataTable(id="table-content", page_size=10),
    ]

    @callback(
        Output("eval-method-description", "children"),
        Input("dropdown-selection", "value"),
    )
    def updateEvalMethod(fieldName: str):
        return f"Evaluation method used: {resultsPerField[fieldName]['method']}"

    @callback(Output("table-content", "data"), Input("dropdown-selection", "value"))
    def updatePerFieldResultTable(fieldName: str):
        return resultsPerField[fieldName]["table"].to_dict("records")

    # these functions are globally used thanks to their callback decorator
    updateEvalMethod = updateEvalMethod
    updatePerFieldResultTable = updatePerFieldResultTable


def run_display_server():
    global _app
    global _is_display_server_running

    if not _is_display_server_running:
        _app.run()
