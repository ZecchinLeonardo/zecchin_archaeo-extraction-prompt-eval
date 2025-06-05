import json
from pathlib import Path
from typing import Iterable, Tuple, cast

from .magoh_target import MagohData, toMagohData
from .models.main_pipeline import ExtractedInterventionData


def save_outputs(outputs: Iterable[Tuple[MagohData, ExtractedInterventionData, float]]):
    def toDict(ex: MagohData, pred: ExtractedInterventionData, score: float):
        answer = toMagohData(pred)
        answer["scheda-intervento"]["id"] = ex["scheda-intervento"]["id"]
        answer = cast(dict, answer)
        answer["score"] = score
        return answer

    toBeSaved = list(map(lambda t: (t[0], toDict(*t)), outputs))
    with Path("./outputs/predicted_answers.json").open("w") as f:
        json.dump(list(map(lambda t: t[1], toBeSaved)), f)
    with Path("./outputs/original_answers.json").open("w") as f:
        json.dump(list(map(lambda t: t[0], toBeSaved)), f)
