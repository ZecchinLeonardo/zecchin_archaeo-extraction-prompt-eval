# Opinions

## Structure of the code

The repository follows a structure of the code which is strongly inspired by
the [*Cookiecutter Data Science*
convention](https://cookiecutter-data-science.drivendata.org/#directory-structure).
Please visit, their website to understand it.

The main Python source code is defined in the `archaeo_super_prompt` poetry
package (in the `src/` subdirectory). It is divided in submodules with those
features:

- the managing of the settings for the remote models, the database and the
tracing server
- some utils for handling the paths in this repository and the caching of the
outputs of the models
- the definition of the extraction models, that will be described in the next
part
- the visualization and tracing functions

## The extraction models

The `archaeo_super_prompt` module relies on custom `scikit-learn` Transformers
and Estimators which can be combined in a Directed-Acyclic Graph (DAG) built
with the `skdag` library, so the following tasks can be achieved within modular
and sometimes scorable and optimizable submodels supporting the standard
`fit`/`transform`/`predict` API:

1. The ingesting of PDF documents related to an archaeological intervention,
with the `VLLM_Preprocessing`
2. The Named-Entity-Recognition/fuzzy-search analyzing of the chunks to try to
pre-select parts of interest of the PDF documents before questioning the LLM
for predicting the value of a Magoh's field. This is achieved with lining the
`NerModel` Transformer with `NeSelector` transformers.
3. The LLM prompting for extracting a given field from chunks of one PDF. This
extractor is a class inheriting the `FieldExtractor` abstract Estimator,
powered by DSPy to enable an automatic optimization of the prompt thanks to
Examples

The building of a DAG instead of a classical scikit-learn Pipeline enables to
handle and show the causal dependencies between some predictors, since some
fields can be guessed from known heuristics including the predicted values of
other fields.

Finally, the whole framework relies on the union of features of Pandas
dataframes.

### Model roles in the pipeline

- **Vision/OCR and embeddings** – The preprocessing step `VLLM_Preprocessing`
  calls the vision model `ibm-granite/granite-vision-3.3-2b` and produces text
  embeddings with `nomic-ai/nomic-embed-text-v1.5` as configured in the
  training DAG【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L42-L55】.
- **Field extraction LLMs** – Extractors such as
  `InterventionStartExtractor` and `ComuneExtractor` query
  `google/gemma-3-27b-it` with a temperature of `0.05`, also defined in the
  training DAG【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L42-L45】【F:prompt_enhancing/src/archaeo_super_prompt/modeling/train.py†L84-L90】.
- **Named‑entity recognition** – A FastAPI service wraps the
  `DeepMount00/Italian_NER_XXL` transformer to supply entity tags for chunk
  filtering【F:prompt_enhancing/models/custom-remote-models/src/magoh_ai_sup_server/inference.py†L8-L18】.

### Supported model providers

- **OpenAI** – `gpt-4.1` via `get_openai_model`, though it is not invoked in
  the current pipeline【F:prompt_enhancing/src/archaeo_super_prompt/modeling/struct_extract/language_model.py†L8-L24】.
- **Ollama** – hosts `granite3.2-vision:latest`, `nomic-embed-text:latest`,
  and `gemma3:27b`【F:prompt_enhancing/models/ollama-models/README.md†L1-L13】.
- **vLLM** – serves `ibm-granite/granite-vision-3.3-2b` and
  `google/gemma-3-27b-it`; embedding model support is planned【F:prompt_enhancing/models/vllm-models/README.md†L1-L11】.
- **Custom NER service** – exposes `DeepMount00/Italian_NER_XXL` through a
  dedicated API【F:prompt_enhancing/models/custom-remote-models/src/magoh_ai_sup_server/inference.py†L8-L18】.
