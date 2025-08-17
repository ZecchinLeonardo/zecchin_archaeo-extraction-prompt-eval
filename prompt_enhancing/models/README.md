# Production models

This directory contains code and untracked files for making already-trained
models ready to be used by client programs like the pipeline.

The usage code directly in the pipeline is programmed in the
`archaeo_super_prompt` main poetry module, not in this directory.

## List of used models

- Vision Large Language Model for the OCR (with ollama or vllm)
- Large Language Model for the data extraction (with ollama or vllm)
- A named-entity recognition model (see the `custom-remote-models/README.md`
file to set it)
