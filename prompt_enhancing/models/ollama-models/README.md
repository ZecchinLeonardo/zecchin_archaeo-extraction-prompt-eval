# Open-source LLM models with Ollama

We propose here a justfile to run an Ollama remote server through ssh commands. This server will serve a part of the LLM models used through the pipeline.

## Configuration

You need to fill in this file a `.env` file as in the `.env.example`.

## Used models

- Vision Language Model for OCR and PDF text reading: `granite3.2-vision:latest`
- Embedding Model: <!-- TODO: figure it out again -->
- Main Language Model for data extraction : `gemma3:27b`
