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
