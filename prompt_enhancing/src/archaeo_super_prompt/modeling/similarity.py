import dspy
from ..config.env import getenv

ollama_localhost_port = 8007 # TODO: change that

def get_ollama_embedding_model(model_id="nomic-embed-text",
                               batch_size=100):
    """Return a dspy client for an Embedding model from the ollama server."""
    return dspy.Embedder(
        f"ollama/{model_id}",
        api_base=getenv("OLLAMA_SERVER_BASE_URL", "http://localhost:11434"),
        api_key="",
        batch_size=batch_size,
    )

def get_vllm_embedding_model(model_id="nomic-embed-text",
                               batch_size=100):
    """Return a dspy client for an Embedding model from the vllm server."""
    return dspy.Embedder(
        f"openai/{model_id}",
        api_base=getenv("OLLAMA_SERVER_BASE_URL", "http://localhost:11434"),
        api_key="",
        batch_size=batch_size,
    )

# In jupyter notebook, be sure to run the line
#    nest_asyncio.apply()
#  as the Embedder forward call use asyncio.run
#  which is not allowed in a IPython kernel which has
#  already its event loop busy
def embed(text: str, embedder: dspy.Embedder):
    return embedder([text])
