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
