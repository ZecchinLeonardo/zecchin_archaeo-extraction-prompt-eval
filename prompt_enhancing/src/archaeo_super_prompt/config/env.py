"""Secret variables management."""
import os
from dotenv import load_dotenv

load_dotenv()


def getenv_or_throw(var_name: str):
    """Load an environment value from the .env file.

    The program will crash if this values does not exist.
    """
    env_var_value = os.getenv(var_name)
    if env_var_value is None:
        raise Exception(
            f"Environment variable '{var_name}' not set up in the .env file"
        )
    return env_var_value
