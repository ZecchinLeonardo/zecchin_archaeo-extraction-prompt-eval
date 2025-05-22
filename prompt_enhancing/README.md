# DSPy extraction data pipeline for assisted prompt enhancing

## Run the code

### Requirements

- `poetry` (>2.1)
- `just` if you are lazy with the commands
- an API key in [OpenRouter](https://openrouter.ai) (50 prompts per day with
the free offer over a set of large language models, 1000 per day with a single
bill of 10$)

### Install dependencies

```sh
poetry install
```

### Setup the environment

You must set in a `.env` file located in the same directory as this README the
following secret environment variable

```sh
OPENROUTER_API_KEY='<your-secret-api-key>'
```

The sample documents must also be loaded in the [dedicated sample documents
directory](../sample_docs/).

### Test some prompting

```sh
just run_main
```
