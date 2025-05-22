"""Module to load the language model provider from OpenRouter
"""

import dspy
import os

def load_model():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable \'OPENROUTER_API_KEY\' not set up in the .env file")

    return dspy.LM(
        "openrouter/meta-llama/llama-3.3-8b-instruct:free",
        api_key=api_key,
    )
