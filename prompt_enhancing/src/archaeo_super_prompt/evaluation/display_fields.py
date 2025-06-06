from pathlib import Path
from typing import Dict, List, Tuple, cast
import mlflow
import pandas as pd

from ..magoh_target import MagohData, toMagohData
from ..models.main_pipeline import ExtractedInterventionData

dfs: List[Tuple[int, Dict[str, pd.DataFrame]]] = []


def highlight_row(row):
    color = "green" if row["validated"] else "red"
    return [f"background-color: {color}"] * (len(row) - 1) + [
        ""
    ]  # skip styling the flag column


def add_dataframes_to_artififact(pack: Dict[str, pd.DataFrame], html_output_path: Path, run_id: str):
    html_parts = []
    for title in pack:
        df = pack[title]
        styled = df.style.apply(highlight_row, axis=1)
        html_parts.append(styled.to_html(caption=title))
    with html_output_path.open("w") as fp:
        full_html = (
            '<html><head><meta charset="utf-8"/></head><body>'
            + "\n".join(html_parts)
            + "</body></html>"
        )
        fp.write(full_html)
    mlflow.log_artifact(str(html_output_path), run_id=run_id)


def add_to_arrays(
    answer: MagohData,
    pred: ExtractedInterventionData,
    metric_values: Dict[str, Dict[str, bool]],
    run: mlflow.ActiveRun
):
    global dfs
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

    # add to the registered dataframes
    dfs.append((scheda_id, pack))

    # add the dataframe in the artifacts
    add_dataframes_to_artififact(pack, Path(f"./outputs/array_{scheda_id}.html"), run.info.run_id)


def score_fields(run: mlflow.ActiveRun):
    global dfs
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
    for k in scores:
        for field_name, validated in cast(pd.Series, (scores[k]["validated"])).items():
            mlflow.log_metric(str(field_name), validated,
                              run_id=run.info.run_id)
    add_dataframes_to_artififact(scores, Path("./outputs/field_scores.html"),
                                 run.info.run_id)


def export_array():
    global dfs
    return dfs
