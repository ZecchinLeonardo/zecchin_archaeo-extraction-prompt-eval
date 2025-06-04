import argparse
from pathlib import Path
import dotenv
import json
import mlflow
from typing import List, cast

from archaeo_super_prompt.inspection.cost import inspect_cost
from archaeo_super_prompt.magoh_target import toMagohData

from .debug_log import print_debug_log, print_warning, set_debug_mode, print_log
from .language_model import load_model
from .open_with_ocr import does_the_content_contains_text, get_all_samples_files, get_report_input
from .models.main_pipeline import ExtractDataFromInterventionReport

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
    set_debug_mode(True)
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
    mlflow.dspy.autolog() #type: ignore
    mlflow.set_experiment("Trying prompts")
    print_log("Tracing ready!\n")

    costs: List[float] = [] # TODO: check if it is not the cost evolution
    
    for input_file_path in get_all_samples_files(input_file_dir_path):
        print_debug_log("Loading sample document...")
        text = get_report_input(input_file_path)
        if not does_the_content_contains_text(text):
            print_warning(f"The content does not contains enough text for file {input_file_path}. Passing it")
            continue
        print_debug_log("Input for the model ready!\n")

        print_debug_log("Prompting and awaiting the parsed answer...")
        response = module.forward_and_type(document_ocr_scan=text)
        print_debug_log("Answer ready!")

        cost = inspect_cost(llm)
        print_log(f"This analyze has approximately cost US${cost}")
        costs.append(cost)

        if response is None:
            continue
        print_debug_log("Transforming into structured data for Magoh...")
        with Path(f"./outputs/{input_file_path.stem}.prediction.json").resolve().open("w") as json_f:
            json.dump(toMagohData(response), json_f)
