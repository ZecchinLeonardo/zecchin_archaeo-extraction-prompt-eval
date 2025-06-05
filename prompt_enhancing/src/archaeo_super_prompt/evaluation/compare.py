import random
from typing import Union, cast
from dspy import Example, Prediction
from dspy.evaluate.metrics import answer_exact_match

from ..magoh_target import MagohData

from ..models.main_pipeline import ExtractedInterventionData

MIN_VALUE = 0
MAX_VALUE = 1

def _validated_json(
    answer: MagohData, pred: ExtractedInterventionData, trace=None
) -> Union[float, bool]:
    # TODO:
    return bool(random.randint(0, 1)) if trace is not None else random.random()


def validated_json(
    example: Example, pred: Prediction, trace=None
) -> Union[float, bool]:
    pred_dict = pred.toDict()
    if not pred_dict:
        return MIN_VALUE if trace is None else False
    return _validated_json(
        cast(MagohData, example.answer),
        cast(ExtractedInterventionData, pred_dict),
        trace,
    )
