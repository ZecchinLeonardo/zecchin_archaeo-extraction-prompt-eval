# 📂 Notebooks experiments 🔬

## 📦 Requirements

To run the notebooks, it is expected that you have already the **Jupyter
stack** in an *external environment*. Indeed, the dev-dependencies of this
Poetry project do not provide this stack, but a way to get a kernel which can
import the project's Python modules in the notebooks for running the
experiments.

### Mandatory requirements

- The following requirements in this Jupyter stack are mandatory :
  - **(jupyter) notebook** or **jupyterlab**
  - the python package [**`poetry-kernel`**](https://github.com/pathbird/poetry-kernel)
- In this project, the development dependencies must also be installed:
  
  ```sh
  poetry install --all-extras
  ```

### Advised dependencies for development

- The following requirements are also recommended (but optional) during the
developments of these notebooks for the versioning to be better:
  - `nbdime`
  - `nbstripout`

You can then run once inside the project

```sh
# source the environment of your jupyter stack, then run the command below
just init-nb-git-workspace
```

## ⚡ Run the notebooks

1. Serve the `notebooks/` directory inside a `jupyter notebook` or `jupyter lab`
run inside your Jupyter stack environment.

  ```sh
  # run one of the commands below
  just run-notebook
  just run-lab
  ```

2. Select the ***Poetry*** kernel when you want to run a notebook. This kernel
is initiated by the `poetry-kernel` module and automatically load the
environment of this Poetry project, including its source modules.
