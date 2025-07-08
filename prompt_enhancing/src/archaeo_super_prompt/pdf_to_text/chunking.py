from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from docling.chunking import HybridChunker

from .types import CorrectlyConvertedDocument

EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

def get_chunker(embed_model_id: str):
    # the tokenizer must be the same as the embedding model
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
    )
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True
    )

def get_chunks(chunker: HybridChunker, document: CorrectlyConvertedDocument):
    chunks = list(chunker.chunk(document.document))
    return chunks
