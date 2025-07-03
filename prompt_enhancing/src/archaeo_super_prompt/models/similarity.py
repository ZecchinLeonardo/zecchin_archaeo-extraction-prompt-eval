import dspy
from ..env import getenv_or_throw

ollama_localhost_port = getenv_or_throw("LOCAL_LLM_PORT")

# TOOD: set this as a model parametre
embedder = dspy.Embedder(
    "ollama/nomic-embed-text",
    api_base=f"http://localhost:{ollama_localhost_port}",
    api_key="",
    batch_size=100,
)

# In jupyter notebook, be sure to run the line
#    nest_asyncio.apply()
#  as the Embedder forward call use asyncio.run
#  which is not allowed in a IKernel which has
#  already its event loop busy
def embed(text: str):
    return embedder([text])
