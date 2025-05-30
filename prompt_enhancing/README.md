# DSPy extraction data pipeline for assisted prompt enhancing

## Run the code

### Requirements

- `poetry` (>=2.1)
- `just` if you are lazy with the commands
- an API key in the OpenAI platform

### Install dependencies

```sh
poetry install
```

### Set up the environment

#### Using your own remote OLlama model

To use a remote cluster, the `justfile` coupled with the `.env` enable to
connect to an ollama server within a ssh tunnel

1. Fill in a `.env` file with the credentials of your cluster, as in the
   part 2 of the `.env.example` file

2. Start the server remotely

   ```sh
   just start-ollama
   ```

3. Launch the tunnel :

   ```sh
   just connect_remote_llm
   ```

   Then you can run your prompts by following the instructions in the next section.

4. When you have finished, stop the tunnel and stop the ollama server with
   running this command

   ```sh
   just stop-ollama
   ```

#### Using an OpenAPI remote model

You must set in a `.env` file located in the same directory as this README the
following secret environment variable

```sh
OPENAI_API_KEY='<your-secret-api-key>'
```

### Test some prompting

Specify an input file in the CLI argument:

```sh
poetry run main --report-dir ../sample_docs/
```

The OCR results and the prompts results are saved in files in the `output/`
directory (please be careful to copy them before a next other run).

File `../sample_docs/Scheda_Intervento_35012` is automatically input in the
`justfile`. Then, you can also test with this shorter command:

```sh
just run_main
```

### Inspect the experiments with MLflow

The traces of the exchanges with the LLM can be viewed in a user-friendly
interface thanks to the Mlflow tracing. These traces are located in a `mlruns/`
untracked directory.

To view them in the local interface, run

```sh
just trace
```

Next, inspect them on `http://localhost:5000`, in the `DSPy > Traces`
tab.
