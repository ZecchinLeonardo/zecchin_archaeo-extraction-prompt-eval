# 📝 LLM and prompt engineering evaluation for Information Extraction

To feed an open database of archeological legacy interventions from a
raw-document dataset, a pipeline with the use of an LLM for the extraction of
structured information is wanted.

This repository is here to evaluate the accuracy of several prompt lists over
several models to achieve this task. It will also allow identifying which NLP
layers should be added in the pipeline to enhance the correctness.

The definition of the accuracy metrics will also be discussed in documentation
in this repository.

## 📂 Structure of the repository

### Prompting enhancement in the pipeline

Source location : [`./prompt_enhancing/`](./prompt_enhancing/)

The prompt engineering work to generate the prompt lists to be tested will be
achieved in this subproject using the [`DSPy` framework](https://dspy.ai/).

### Benchmarking the prompt lists over several models

Source location : [`./benchmarking/`](./benchmarking/)

To evaluate all the prompt lists over several models, the [`promptfoo`
library](https://www.promptfoo.dev/) will be used.
