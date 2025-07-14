import os
from dotenv import load_dotenv

load_dotenv()


def getenv_or_throw(var_name: str):
    env_var_value = os.getenv(var_name)
    if env_var_value is None:
        raise Exception(
            f"Environment variable '{var_name}' not set up in the .env file"
        )
    return env_var_value
