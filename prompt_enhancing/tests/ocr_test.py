from pathlib import Path
from archaeo_super_prompt.open_with_ocr import pdf_to_text, normalize_alpha_words
from archaeo_super_prompt.main import setup

def test_scan_with_ocr_only():
    got_text = pdf_to_text(Path("../sample_docs/Scheda_Intervento_37724.pdf").resolve())
    assert(normalize_alpha_words(got_text) != ""), "the text is empty!!"
    print(got_text)

if __name__ == "__main__":
    setup()
    test_scan_with_ocr_only()

