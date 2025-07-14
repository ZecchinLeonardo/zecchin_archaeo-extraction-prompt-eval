from typing import cast
import dspy


class Assess(dspy.Signature):
    """Assess a predicted named entity enough matches with with the expected one."""

    expected_entity: str = dspy.InputField()
    predicted_entity: str = dspy.InputField()
    similarity_percentage: float = dspy.OutputField(
        desc="The percentage of similarity between both entities (between 0 and 1) that you have to estimate"
    )

def normalize_percentage(untrustable_percentage: float) -> float:
    positive = abs(untrustable_percentage)
    if 1 < positive <= 100:
        # The llm might have answered a percentage with %
        return positive/100
    if positive <= 1:
        return positive
    return 0 # the llm has wrongly evaluated the similarity


def check_with_LLM(treshold: float):
    assert 0 <= treshold <= 1, "The treshold must be a percentage"

    def check_with_LLM(gold: str, pred: str, trace=None):
        correct = (
            normalize_percentage(
                cast(
                    Assess,
                    dspy.Predict(Assess)(expected_entity=gold, predicted_entity=pred),
                ).similarity_percentage
            )
            >= treshold
        )
        return correct if trace is not None else int(correct)

    return check_with_LLM

class DateAssess(dspy.Signature):
    """Check if two dates representations define the same real date"""
    expected_date_representation: str = dspy.InputField()
    predicted_date_representation: str = dspy.InputField()
    are_they_talking_about_the_same_date: bool = dspy.OutputField()

def check_date_with_LLM(gold: str, pred: str, trace=None):
    correct = cast(
                DateAssess,
                dspy.Predict(DateAssess)(expected_date_representation=gold, predicted_date_representation=pred),
            ).are_they_talking_about_the_same_date
    return correct if trace is not None else int(correct)
