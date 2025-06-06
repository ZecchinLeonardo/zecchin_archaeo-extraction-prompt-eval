import argparse
from pathlib import Path
import dotenv
from dspy import Example, Prediction
import mlflow
from typing import List, Tuple, cast

from archaeo_super_prompt.evaluation.display_fields import export_array, score_fields
from archaeo_super_prompt.evaluation.evaluate import get_evaluator
from archaeo_super_prompt.inspection.cost import inspect_cost
from archaeo_super_prompt.output import save_outputs

from .debug_log import set_debug_mode, print_log
from .language_model import load_model
from .models.main_pipeline import ExtractDataFromInterventionReport, ExtractedInterventionData

def load_file_input_path_from_arg():
    parser = argparse.ArgumentParser(description="The relative path of the directory with all the \"relazione di scava\" to be analyzed (all txt files in).")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Relative filepath of directory of txt files, e.g., \
        '../pdf_to_text/extracted_texts'"
    )
    args = parser.parse_args()
    return Path(cast(str, args.report_dir))

def setup() -> None:
    set_debug_mode(False)
    dotenv.load_dotenv()

def main() -> None:
    setup()

    input_file_dir_path = load_file_input_path_from_arg()

    print_log("Initialising the LLM...")
    llm = load_model()
    print_log("LLM ready to be used!\n")

    print_log("Instanciating the DSPy module...")
    module = ExtractDataFromInterventionReport()
    print_log("DSPy module ready!\n")

    print_log("Instanciating mlflow tracing...")
    mlflow.set_experiment("Meeting (prediction for chunking)")
    mlflow.dspy.autolog(log_evals=True) #type: ignore
    with mlflow.start_run() as run:
        print_log("Tracing ready!\n")
        
        evaluate = get_evaluator(input_file_dir_path, run, return_outputs=True)
        results = cast(Tuple[float, List[Tuple[Example, Prediction, float]]], evaluate(module))

        cost = inspect_cost(llm)
        print_log(f"Cost of this evaluation (in US$): {cost}")

        score_fields(run)
        save_outputs(((ex.answer, cast(ExtractedInterventionData, pred.toDict()), score) for ex, pred, score in filter(lambda t: t[1].toDict(), results[1])))

        for scheda_id, d in export_array():
            print("-"*5, scheda_id, "-"*5)
            for t in d:
                print(t)
                print(d[t], "\n")
            print("-"*5, scheda_id, "-"*5, "\n")
