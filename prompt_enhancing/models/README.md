# Production models

This directory contains code and untracked files for making already-trained
models ready to be used by client programs like the pipeline.

The usage code directly in the pipeline is programmed in the
`archaeo_super_prompt` main poetry module, not in this directory.

## List of used models

- Vision Large Language Model for the OCR
- Tokenizer Model
- Embedding Model for query-content similarity estimation. The previous
Tokenizer must be set [according to
it](https://docling-project.github.io/docling/examples/hybrid_chunking/#basic-usage)
- Large Language Model for the data extraction
