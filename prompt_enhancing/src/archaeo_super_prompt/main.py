import argparse
import dotenv
import dspy
from typing import cast

from .debug_log import set_debug_mode, print_log
from .language_model import load_model
from .open_with_ocr import pdf_to_text, save_log_in_file
from .models.main_pipeline import ExtractDataFromInterventionReport

def load_file_input_path_from_arg():
    parser = argparse.ArgumentParser(description="The relative path of the \"relazione di scava\" document you want to analyze.")
    parser.add_argument(
        "--report",
        required=True,
        help="PDF relative filepath, e.g., '../sample_docs/Scheda_Intervento_35012.pdf'"
    )
    args = parser.parse_args()
    return cast(str, args.report)

def main():
    set_debug_mode(True)
    dotenv.load_dotenv()
    input_file_path = load_file_input_path_from_arg()

    print_log("Initialising the LLM...")
    load_model()
    print_log("LLM ready to be used!")

    print_log("Instanciating the DSPy module...")
    """For now, this simple module with such a complex signature is not
    suitable for the llm and the randomness of its outputs, which make the
    pipeline and the sending of others dspy automatic prompts fail.
    We do need to create a more complex module which better target the
    predicition of the fields to be sure they are type-safe.

    Typesafety must be guaranteed by the model
    """
    module = ExtractDataFromInterventionReport()
    print_log("DSPy module ready!")

    print_log("Loading sample document...")
    text = pdf_to_text(input_file_path)
    save_log_in_file("./outputs/ocr.txt", text)
    print_log("Document converted into text!")

    print_log("Prompting and awaiting the parsed answer...")
    response = module(document_ocr_scan=text)
    print_log("Answer ready:")
    print(response)
    save_log_in_file("./outputs/prediction.txt", str(response))

    # TODO: use MLflow instead
    dspy.inspect_history(n=10)

    # TODO:
    # print_log("Transforming into structured data for Magoh...")
    # print(toMagohData(response))
