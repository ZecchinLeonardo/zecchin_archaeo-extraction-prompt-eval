from pathlib import Path
from typing import Dict, List, Tuple
import mlflow
import pandas as pd

from ..magoh_target import MagohData, toMagohData
from ..models.main_pipeline import ExtractedInterventionData

dfs: List[Tuple[int, Dict[str, pd.DataFrame]]] = []


def add_to_arrays(
    answer: MagohData,
    pred: ExtractedInterventionData,
    metric_values: Dict[str, Dict[str, bool]],
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
    html_parts = []
    for title in pack:
        df = pack[title]
        styled = df.style.apply(highlight_row, axis=1)
        html_parts.append(styled.to_html(caption=title))
    html_output_path = Path(f"./outputs/array_{scheda_id}.html")
    with html_output_path.open("w") as fp:
        full_html = "<html><head><meta charset=\"utf-8\"/></head><body>" + "\n".join(html_parts) + "</body></html>"
        fp.write(full_html)
    mlflow.log_artifact(str(html_output_path))


def highlight_row(row):
    color = "green" if row["validated"] else "red"
    return [f"background-color: {color}"] * (len(row) - 1) + [
        ""
    ]  # skip styling the flag column

# TODO:
# def score_fields():
#     global dfs
#     scores = {}
#     d = dfs[0][1]
#     for title in d:
#         scores[title] = {}
#         keys = d.keys()
#         for key in keys:
#             scores[title][key] += 0
#             for i in range(len(dfs)):
#                 scores[title][key] += dfs[i][1].at[title, key]


def export_array():
    global dfs
    return dfs
