import argparse
from pathlib import Path
import dotenv
import json
import mlflow
from typing import List, cast

from archaeo_super_prompt.inspection.cost import inspect_cost
from archaeo_super_prompt.magoh_target import toMagohData

from .debug_log import set_debug_mode, print_log
from .language_model import load_model
from .open_with_ocr import get_all_samples_files, init_ocr_setup, normalize_alpha_words, pdf_to_text, save_log_in_file
from .models.main_pipeline import ExtractDataFromInterventionReport

def load_file_input_path_from_arg():
    parser = argparse.ArgumentParser(description="The relative path of the directory with all the \"relazione di scava\" to be analyzed (all pdf files in).")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Relative filepath of directory of PDF files, e.g., '../sample_docs/'"
    )
    args = parser.parse_args()
    return Path(cast(str, args.report_dir))

def setup() -> None:
    set_debug_mode(True)
    dotenv.load_dotenv()
    init_ocr_setup()

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
    mlflow.set_experiment("First prompts")
    print_log("Instance created!\n")

    costs: List[float] = [] # TODO: check if it is not the cost evolution
    
    for input_file_path in get_all_samples_files(input_file_dir_path):
        print_log("Loading sample document...")
        text = pdf_to_text(input_file_path)
        assert(text != ""), "OCR result is empty ??"
        print(len(normalize_alpha_words(text)))
        print_log("Document converted into text!\n")
        # save_log_in_file(f"./outputs/{input_file_path.name}.ocr.txt", text)
        print_log("Prompting and awaiting the parsed answer...")
        response = module.forward_and_type(document_ocr_scan=text)
        print_log("Answer ready:")
        print(response, "\n")

        cost = inspect_cost(llm)
        print_log(f"This analyze has approximately cost US${cost}")
        costs.append(cost)

        print_log("Transforming into structured data for Magoh...")
        with Path(f"./outputs/{input_file_path.name}.prediction.json").resolve().open("w") as json_f:
            json.dump(toMagohData(response), json_f)
