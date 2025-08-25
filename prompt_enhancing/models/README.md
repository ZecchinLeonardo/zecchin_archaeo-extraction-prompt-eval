# Production models

This directory contains code and untracked files for making already-trained
models ready to be used by client programs like the pipeline.

The usage code directly in the pipeline is programmed in the
`archaeo_super_prompt` main poetry module, not in this directory.

## Models used in the pipeline

- **Vision/OCR** – `ibm-granite/granite-vision-3.3-2b` invoked by
  `VLLM_Preprocessing` during PDF ingestion【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L42-L55】.
- **Embedding** – `nomic-ai/nomic-embed-text-v1.5` computed in the same
  preprocessing stage【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L42-L55】.
- **Field extraction LLM** – `google/gemma-3-27b-it` (temperature `0.05`) used
  by the `InterventionStartExtractor` and `ComuneExtractor` components【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L42-L45】【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L84-L90】.
- **Named‑entity recognition** – `DeepMount00/Italian_NER_XXL` served through
  a FastAPI wrapper【F:prompt_enhancing/models/custom-remote-models/src/magoh_ai_sup_server/inference.py†L8-L18】.

## Available model providers

- **OpenAI** – `gpt-4.1` via `get_openai_model`, currently unused in the
  notebooks【F:prompt_enhancing/src/archaeo_super_prompt/modeling/struct_extract/language_model.py†L8-L24】.
- **Ollama** – local hosting for `granite3.2-vision:latest`,
  `nomic-embed-text:latest`, and `gemma3:27b`【F:prompt_enhancing/models/ollama-models/README.md†L1-L13】.
- **vLLM** – serves `ibm-granite/granite-vision-3.3-2b` and
  `google/gemma-3-27b-it`, with embedding support on the roadmap【F:prompt_enhancing/models/vllm-models/README.md†L1-L11】.
- **Custom NER service** – exposes `DeepMount00/Italian_NER_XXL` for entity
  tagging【F:prompt_enhancing/models/custom-remote-models/src/magoh_ai_sup_server/inference.py†L8-L18】.
