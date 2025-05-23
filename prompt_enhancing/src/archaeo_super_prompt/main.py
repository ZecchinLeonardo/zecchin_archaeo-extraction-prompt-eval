import argparse
import dotenv
import dspy
from typing import cast

from .debug_log import set_debug_mode, print_log
from .language_model import load_model
from .magoh_target import toMagohData
from .open_with_ocr import pdf_to_text
from .signature.arch_extract_type import ArchaeologicalInterventionData

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
    lm = load_model()
    dspy.configure(lm=lm)
    print_log("LLM ready to be used!")

    print_log("Instanciating the DSPy module...")
    module = dspy.ChainOfThought(ArchaeologicalInterventionData)
    print_log("DSPy module ready!")

    print_log("Loading sample document...")
    text = pdf_to_text(input_file_path)
    print_log("Document converted into text!")

    print_log("Prompting and awaiting the parsed answer...")
    CONTEXT = """You are analysing an Italian official document about an archaeological intervention and you are going to extract in Italian some information as the archivists in archeology do.
Some information are optional as a document can forget to mention it, then try to think if you can figure it out or if you have to answer nothing for these field."""
    response: ArchaeologicalInterventionData = module(context=CONTEXT, italian_archaeological_document=text)
    print_log("Answer ready:")
    print(response)

    print_log("Transforming into structured data for Magoh...")
    print(toMagohData(response))
