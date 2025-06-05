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

The training set must be a directory with a structure like this:

```sh
inputs/
├── 35012 # each id in the sample_answers.json must have its directory
│   ├── Scheda_Intervento_35012.pdf # pdf for now are not used
│   └── Scheda_Intervento_35012.txt # extracted text file mandatory
├── 37084
│   ├── Relazione_di_scava.pdf # file names are not an issue
│   ├── Relazione_di_scava__scanned.txt # file names are not an issue
│   └── Additional_document.txt # several sources can be given
...
├── 37822
│   ├── Scheda_Intervento_37822.pdf
│   └── Scheda_Intervento_37822.txt
└── sample_answers.json
```

Specify the directory containing the training set in the CLI argument:

```sh
poetry run main --report-dir ./inputs/
# or
just run_main ./inputs/
```

The prompts results are saved in json files in the `output/` directory (please
be careful to copy them before a next other run, if you want to save them).

The directory `./inputs/` is by default given in arguments in the `justfile`.
Then, you can also test with this shorter command:

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
