import argparse
from pathlib import Path
import dotenv
from dspy import Example, Prediction
import dspy
import mlflow
from typing import List, Tuple, cast

from archaeo_super_prompt.evaluation.evaluate import get_evaluator
from archaeo_super_prompt.inspection.cost import inspect_cost
from archaeo_super_prompt.output import save_outputs

from .debug_log import set_debug_mode, print_log
from .language_model import load_model
from .models.main_pipeline import ExtractDataFromInterventionReport, ExtractedInterventionData

def load_file_input_path_from_arg():
    parser = argparse.ArgumentParser(description="Run structured data extraction evaluation over the given evaluation dataset")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Relative filepath of directory of txt files, e.g., \
        '../pdf_to_text/extracted_texts'"
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="The name of the experiment in mlflow"
    )
    args = parser.parse_args()
    return Path(cast(str, args.report_dir)), cast(str, args.experiment_name)

def setup() -> None:
    set_debug_mode(False)
    dotenv.load_dotenv()

def main() -> None:
    setup()

    input_file_dir_path, exp_name = load_file_input_path_from_arg()

    print_log("Initialising the LLM...")

    llms = { temp_x20: load_model(temp_x20/20) for temp_x20 in range(9)}
    evaluation_llm = llms[0]
    dspy.configure(lm=evaluation_llm)

    print_log("LLM ready to be used!\n")

    print_log("Instanciating the DSPy module...")
    module = ExtractDataFromInterventionReport()
    print_log("DSPy module ready!\n")

    print_log("Instanciating mlflow tracing...")
    mlflow.set_experiment(exp_name)

    mlflow.dspy.autolog(log_evals=True) #type: ignore
    evaluate = get_evaluator(input_file_dir_path, return_outputs=True)
    print_log("Tracing ready!\n")
    with mlflow.start_run():
        for temperature, llm in llms.items():
            module.set_lm(llm)
            with mlflow.start_run(run_name=f"temp={temperature}", nested=True) as active_run:
            
                results = cast(Tuple[float, List[Tuple[Example, Prediction, float]]], evaluate(module, active_run))

                cost = inspect_cost(llm)
                print_log(f"Cost of this evaluation (in US$): {cost}")

                print_log("Saving outputs")
                save_outputs(((ex.answer, cast(ExtractedInterventionData, pred.toDict()), score) for ex, pred, score in filter(lambda t: t[1].toDict(), results[1])))
