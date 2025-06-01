# Any pdf to text

This part of the repository contains all the necessary documentation and code
to run a strong third-party pdf scanner, able to do the following tasks :

1. Transforming PDF documents by adding an OCR-red text layer to them, with
   optimizing the reading of italian-written documents. 
2. Chunking the finally-directly extractable text of PDF documents according to
   its human-viewable layout.

## OCR layer

The stack used is ocrmypdf, which underneath use tesseract with the italian
trained data. It is in a docker that can be build and run as a standalone
application with these `just` shortcuts

```sh
just build
```

```sh
just ocr my_file.pdf my_output.pdf
```

## Layout chunking

We use the
[LayoutPDFReader](https://github.com/nlmatics/llmsherpa?tab=readme-ov-file#layoutpdfreader)
AI-powered tool for chunking in an optimized way the extractable text of a
document, with keeping the text information stored in list and tables
in a format cleanly structured for AI understanding. 

We will show here also how to self-host the llmsherpa server required to do so.
