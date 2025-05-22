import dspy
import dotenv

from .language_model import load_model

from .open_with_ocr import pdf_to_text
from .signature import ExtractionArcheoData


def main():
    dotenv.load_dotenv()
    lm = load_model()
    dspy.configure(lm=lm)
    module = dspy.Predict(ExtractionArcheoData)
    text = pdf_to_text("../sample_docs/Scheda_Intervento_37640.pdf")
    response = module(text=text)
    print(response)
