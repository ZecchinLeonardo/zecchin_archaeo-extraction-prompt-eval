from llmsherpa.readers import Block, LayoutPDFReader
from pathlib import Path

from ..cache import memory
from ..signatures.input import ExtractedPDFContent

# TODO: edit it with .env
_llm_sherpa_api_url = (
    "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=yes"
)
_pdf_reader = LayoutPDFReader(_llm_sherpa_api_url)


@memory.cache
def extract_smart_chunks_from_pdf(pdf_path: Path) -> ExtractedPDFContent:
    doc = _pdf_reader.read_pdf(str(pdf_path))
    chunks: list[Block] = doc.chunks()
    if not chunks:
        raise Exception("No content to be extracted")
    total_page_number = max(chunk.page_idx for chunk in chunks)
    content: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        try:
            chunk_ctx_content = chunk.to_context_text()
            semantic_chunk_info = f"Chunk {i} ({chunk.tag} page \
{chunk.page_idx}/{total_page_number})"
            content[semantic_chunk_info] = chunk_ctx_content
        except *Exception:
            pass
    return content
