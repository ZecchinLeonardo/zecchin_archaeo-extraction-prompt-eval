import dspy
from pathlib import Path

from archaeo_super_prompt.evaluation.compare import validated_json

from .load_examples import load_examples

def get_evaluator(input_file_dir_path: Path, return_outputs=False):
    devset = load_examples(input_file_dir_path)
    metric = validated_json
    # TODO: parametrize some settings
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        return_outputs=return_outputs,
        provide_traceback=True, # TODO: remove it for traceback
        num_threads=1,
        display_progress=True,
        display_table=5,
    )
    def evaluate(program: dspy.Module):
        return evaluator(program)
    return evaluate

