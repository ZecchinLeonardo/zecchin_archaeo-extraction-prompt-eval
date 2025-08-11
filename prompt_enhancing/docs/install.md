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

## 3. Raw data to install

TODO
