"""Module to load the language model provider from OpenRouter
"""

import dspy
import os

def load_model():
    """Configure dspy to load an internally chosen LLM.
    The OPENAI_API_KEY environement variable must be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise EnvironmentError("Environment variable \'OPENAI_API_KEY\' not set up in the .env file")

    analysing_model = dspy.LM(
        "openai/gpt-4.1",
        api_key=api_key,
    )
    dspy.configure(lm=analysing_model)
