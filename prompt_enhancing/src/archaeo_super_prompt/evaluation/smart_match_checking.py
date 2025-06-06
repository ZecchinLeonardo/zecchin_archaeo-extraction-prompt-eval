from typing import cast
import dspy


class Assess(dspy.Signature):
    """Assess a predicted named entity matches with with the expected one."""

    expected_entity: str = dspy.InputField()
    predicted_entity: str = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

def check_with_LLM(gold: str, pred: str, trace=None):
    correct = cast(Assess, dspy.Predict(Assess)(expected_entity=gold, predicted_entity=pred)).assessment_answer
    return correct if trace is not None else int(correct)
