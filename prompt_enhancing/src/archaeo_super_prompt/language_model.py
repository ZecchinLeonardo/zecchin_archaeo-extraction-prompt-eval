"""Module to load the language model provider from OpenRouter
"""

import dspy
import os

def _get_openai_model():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable \'OPENAI_API_KEY\' not set up in the .env file")

    return dspy.LM(
        "openai/gpt-4.1",
        api_key=api_key,
    )

def _get_ollama_model():
    ollama_localhost_port = os.getenv("LOCAL_LLM_PORT")
    if ollama_localhost_port is None:
        raise EnvironmentError("Environment variable \'LOCAL_LLM_PORT\' not set up in the .env file")
    return dspy.LM(
        "ollama_chat/gemma3",
        api_base=f"http://localhost:{ollama_localhost_port}",
        api_key='',
    )
     
def load_model():
    """Configure dspy to load an internally chosen LLM.
    The OPENAI_API_KEY environement variable must be set.
    """
    analysing_model = _get_ollama_model()
    dspy.configure(lm=analysing_model)
    return analysing_model
