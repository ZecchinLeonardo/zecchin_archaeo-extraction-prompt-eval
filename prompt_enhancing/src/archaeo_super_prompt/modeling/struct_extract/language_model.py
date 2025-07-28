"""Module to load the language model provider from OpenRouter"""

import dspy

from ...config.env import getenv_or_throw


def _get_openai_model():
    api_key = getenv_or_throw("OPENAI_API_KEY")

    return dspy.LM(
        "openai/gpt-4.1",
        api_key=api_key,
    )


def _get_ollama_model(temperature=0.0):
    return dspy.LM(
        "ollama_chat/gemma3:27b",
        api_base="http://localhost:11434",
        api_key="",
        temperature=temperature,
    )

def _get_vllm_model(temperature=0.0):
    return dspy.LM(
        "openai/google/gemma-3-27b-it",
        api_base="http://localhost:8006/v1",
        api_key="",
        temperature=temperature,
    )

def load_model(temperature=0.0):
    """Configure dspy to load an internally chosen LLM.
    The OPENAI_API_KEY environement variable must be set.
    """
    analysing_model = _get_vllm_model(temperature)
    return analysing_model
