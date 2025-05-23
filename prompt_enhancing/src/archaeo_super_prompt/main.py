from .debug_log import set_debug_mode, print_log
import dspy
import dotenv

from .language_model import load_model

from .open_with_ocr import pdf_to_text
from .signature.arch_extract_type import ArchaeologicalInterventionData


def main():
    set_debug_mode(True)
    dotenv.load_dotenv()
    print_log("Initialising the LLM...")
    lm = load_model()
    dspy.configure(lm=lm)
    print_log("LLM ready to be used!")
    print_log("Instanciating the DSPy module...")
    module = dspy.Predict(ArchaeologicalInterventionData)
    print_log("DSPy module ready!")
    print_log("Loading sample document...")
    text = pdf_to_text("../sample_docs/Scheda_Intervento_37640.pdf")
    print_log("Document converted into text!")
    print_log("Prompting and awaiting the parsed answer...")
    CONTEXT = """You are analysing an Italian official document about an archaeological intervention and you are going to extract in Italian some information as the archivists in archeology do.
    
Some information are optional as a document can forget to mention it, then try to think if you can figure it out or if you have to answer nothing for these field."""
    response = module(context=CONTEXT, italian_archaeological_document=text)
    print_log("Answer ready:")
    print(response)
