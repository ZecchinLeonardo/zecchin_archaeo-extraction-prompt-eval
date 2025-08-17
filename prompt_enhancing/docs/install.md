# Installation

We describe in this section all the steps to install a development
workspace for developing with the extraction.

## 1. Environment requirements

In your environment you need the following dependencies (cf. the next section
for a rootless, isolated installation of this environment):

- `python` (>= 3.13)
- `poetry` (>=2.1)
- `graphviz`
- `jupyter-notebook` or `jupyter-lab`, with the python package [**`poetry-kernel`**](https://github.com/pathbird/poetry-kernel)
- `just` for running the main commands

### Recommended setup with conda

The easiest way to get this environment in your user session on your machine is
creating a conda environment from either

- the `environment.yml` file:

  ```sh
  $CONDA env create --prefix ./arch-env -f environment.yml
  ```

- the following command, to get the last versions of the dependencies:

  ```sh
  $CONDA create --prefix ./arch-env python=3.13 poetry graphviz jupyterlab \
    poetry-kernel just nbdime nbstripout
  ```

Then work inside this environment with the following command:

```sh
$CONDA activate ./arch-env
```

## 2. Project's dependencies

```sh
just install
```

For developing with the notebooks, run also this rule (this enables stripout of
the cells' outputs for the git diffs and also the nbdiff):

```sh
just init-nb-git-workspace
```

## 3. Required data

### Remote databases

1. A postgresql with the following tables must be available:
    - featured_intervention_data
    - findings
2. A minio file store must also be available.

To set the credentials to these databases, fill a `.env` file from the
`.env.example` file.

### Local csv files

The following csv files must be present in the the `data/raw` directory

```sh
data/raw/
└── thesaurus
    ├── comune.csv
    └── provincie.csv
```

The keys of these files must be at least the following:

- `comune.csv`: id (`int`), id_com (`int`), nome (`str`), provincia (`int`)
- `provincie.csv`: id_prov (`int`), nome (`str`), sigla (`char[2]`)

## 4. Remote AI models

According to the components you use in the extraction model, you will need AI
models to be run in other processes. The program fetches them through HTTP
requests.

See the `models/README.md` file to set those models.

Once you can connect to them, set those environment variables if needed:

- `NER_MODEL_HOST_URL`
- `VLM_HOST_URL`
- `VLLM_SERVER_BASE_URL` or `OLLAMA_SERVER_BASE_URL` or `OPENAI_API_KEY`
