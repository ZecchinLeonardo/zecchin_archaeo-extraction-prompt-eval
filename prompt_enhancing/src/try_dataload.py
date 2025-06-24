from feature_engine.pipeline import Pipeline

from archaeo_super_prompt.pdf_to_text import OCR_Transformer, TextExtractor

# TODO: better manage the cache
pipeline = Pipeline(
    [
        ("ocr", OCR_Transformer),
        ("pdf_reader", TextExtractor),
        # ("extractor", MagohDataExtractor()),
    ]
)
