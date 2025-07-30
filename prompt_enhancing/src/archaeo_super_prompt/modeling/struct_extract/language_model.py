"""Module to load the language model provider."""

import dspy

from ...config.env import getenv_or_throw, getenv


def get_openai_model(model_id="gpt-4.1", temperature=0.0):
    """Return a dspy language model client bound to the OpenAI API.

    Arguments:
        model_id: the identifier as in the OpenAI api: https://dspy.ai/learn/programming/language_models/
        temperature: the temperature of the model during its usage.

    Environment requirements:
        The OPENAI_API_KEY envrionment variable must be defined to use the API
    """
    api_key = getenv_or_throw("OPENAI_API_KEY")

    return dspy.LM(
        f"openai/{model_id}",
        api_key=api_key,
        temperature=temperature
    )


def get_ollama_model(model_id="gemma3:27b", temperature=0.0):
    """Return a dspy language model client bound to an ollama server.

    Arguments:
        model_id: see this page: https://dspy.ai/learn/programming/language_models/
        temperature: the temperature of the model during its usage.

    Environment requirements:
        The OLLAMA_SERVER_BASE_URL envrionment variable can be defined to
        override the default ollama api's base url, served on http://localhost:11434
    """
    ollama_server_base_url = getenv(
        "OLLAMA_SERVER_BASE_URL", "http://localhost:11434"
    )
    return dspy.LM(
        f"ollama_chat/{model_id}",
        api_base=ollama_server_base_url,
        api_key="",
        temperature=temperature,
    )


def get_vllm_model(model_id="google/gemma-3-27b-it", temperature=0.0):
    """Return a dspy language model client bound to a vllm server.

    Arguments:
        model_id: the identifier of the model as in the hugging face hub; see this page: https://dspy.ai/learn/programming/language_models/
        temperature: the temperature of the model during its usage.

    Environment requirements:
        The VLLM_SERVER_BASE_URL envrionment variable can be defined to
        override the default ollama api's base url, served on http://localhost:8006/v1
    """
    vllm_server_base_url = getenv(
        "VLLM_SERVER_BASE_URL", "http://localhost:8006/v1"
    )
    return dspy.LM(
        f"openai/{model_id}",
        api_base=vllm_server_base_url,
        api_key="",
        temperature=temperature,
    )
