from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import mlflow
import pandas as pd

from archaeo_super_prompt.debug_log import print_debug_log

from ..magoh_target import MagohData, toMagohData
from ..models.main_pipeline import ExtractedInterventionData

"""(magoh record id, labeled set of visual score tables)"""
VisualScoreForMagohRecord = Tuple[int, Dict[str, pd.DataFrame]]

Color = Tuple[int, int, int]

# (Column key, worst color, target color)
DataFrameDisplayData = Tuple[str, Color, Color]

_RED: Color = (248, 130, 130)
_GREEN: Color = (76, 179, 145)
_FADED_GREEN: Color = (246, 252, 250)

def apply_row_gradient(column_key: str, worst_color: Color, target_color: Color):
    def style_to_apply(row):
        factor = float(row[column_key])  # value between 0 and 1

        r = int(worst_color[0] + (target_color[0] - worst_color[0]) * factor)
        g = int(worst_color[1] + (target_color[1] - worst_color[1]) * factor)
        b = int(worst_color[2] + (target_color[2] - worst_color[2]) * factor)
        style = f"background-color: rgb({r},{g},{b})"
        return [style for _ in row]

    return style_to_apply


def add_dataframes_to_artififact(
    pack: Dict[str, pd.DataFrame],
    df_disp_data: DataFrameDisplayData,
    html_output_path: Path,
    run: Optional[mlflow.ActiveRun],
):
    html_parts = []
    for title in pack:
        df = pack[title]
        styled = df.style.apply(apply_row_gradient(*df_disp_data), axis=1)
        html_parts.append(styled.to_html(caption=title))
    with html_output_path.open("w") as fp:
        full_html = (
            '<html><head><meta charset="utf-8"/></head><body>'
            + "\n".join(html_parts)
            + "</body></html>"
        )
        fp.write(full_html)
    if run is not None:
        mlflow.log_artifact(str(html_output_path), run_id=run.info.run_id)


def save_visual_score_table(
    answer: MagohData,
    pred: ExtractedInterventionData,
    metric_values: Dict[str, Dict[str, bool]],
    run: Optional[mlflow.ActiveRun],
) -> VisualScoreForMagohRecord:
    pred_mg = toMagohData(pred)
    scheda_id = answer["scheda-intervento"]["id"]
    df_titles = metric_values.keys()
    pack = {
        k: pd.DataFrame(
            {
                "expected": list(answer[k].values()),
                "predicted": list(pred_mg[k].values()),
                "validated": list(metric_values[k].values()),
            },
            index=list(metric_values[k].keys()),  # type: ignore
        )
        for k in df_titles
    }

    # add the dataframe in the artifacts
    add_dataframes_to_artififact(
        pack,
        ("validated", _RED, _GREEN),
        Path(f"./outputs/array_{scheda_id}.html"),
        run,
    )

    # add to the registered dataframes
    return (scheda_id, pack)


def score_fields(dfs: List[VisualScoreForMagohRecord],
                 run: Optional[mlflow.ActiveRun]):
    print_debug_log(str(dfs))
    if run is not None:
        print_debug_log(f"run id: {run.info.run_id}")
    else:
        print_debug_log("no run")

    if len(dfs) == 0:
        return
    titles = (dfs[0][1]).keys()
    scores = {
        title: cast(
            pd.Series,
            sum(cast(pd.Series, dfs[i][1][title]["validated"]) for i in range(len(dfs)))
            / len(dfs),
        ).to_frame()
        for title in titles
    }

    # Plot metrics score for each field
    if run is not None:
        for k in scores:
            for field_name, validated in cast(pd.Series, (scores[k]["validated"])).items():
                mlflow.log_metric(str(field_name), validated, run_id=run.info.run_id)

    # Save these scores in a dataframe
    add_dataframes_to_artififact(
        scores,
        ("validated", _FADED_GREEN, _GREEN),
        Path("./outputs/field_scores.html"),
        run,
    )
