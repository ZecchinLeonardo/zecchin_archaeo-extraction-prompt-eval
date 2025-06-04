# 📜➡📄 Any PDF to text

This part of the repository contains all the necessary documentation and code
to run a strong standalone pdf scanner, able to do the following tasks :

1. Transforming PDF documents by adding an OCR-read text layer to them, with
   optimizing the reading of italian-written documents.
2. ~~Chunking the finally-directly extractable text of PDF documents according to
   its human-viewable layout.~~ Extracting the extractable text from PDF
   documents.

> The chunking is a feature coming soon.

## Run all the pipeline

```sh
# Produce a text file in `extracted_texts` for each pdf files in `sample_docs`
# Apply an OCR scan, if required.
just extract_text ../sample_docs/ ./extracted_texts/
```

## OCR layer

The stack used is [ocrmypdf](https://ocrmypdf.readthedocs.io/en/latest/), which
underneath use [tesseract](https://tesseract-ocr.github.io/) with the italian
trained data. It is in a docker that can be built and run as a standalone
application with these `just` shortcuts

```sh
just build
```

```sh
# will apply an ocr-read layer on each pdf file in the `inputs` directory
# the new pdf/a files are output here in the `cached_ocrs` directory
just ocr ./inputs/ ./cached_ocrs/
```

## Layout chunking *(TODO)*

> Temporary solution : `pdftotext` of Poppler
> ```sh
> # for each already-ocr-read pdf in `cached_ocrs`, extract the text
> # the extracted text is stored in .txt files in `extracted_texts`
> just batch_layoutpdfreader ./cached_ocrs/ ./extracted_texts/
> ```

~~We use the
[LayoutPDFReader](https://github.com/nlmatics/llmsherpa?tab=readme-ov-file#layoutpdfreader)
AI-powered tool for chunking in an optimized way the extractable text of a
document, with keeping the text information stored in list and tables
in a format cleanly structured for AI understanding.~~

~~We will show here also how to self-host the llmsherpa server required to do
so.~~
