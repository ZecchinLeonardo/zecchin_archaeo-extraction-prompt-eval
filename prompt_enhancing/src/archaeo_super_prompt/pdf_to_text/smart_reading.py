from llmsherpa.readers import Block, LayoutPDFReader
from pathlib import Path

from ..cache import memory
from ..signatures.input import ExtractedPDFContent

# TODO: edit it with .env
_llm_sherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=yes"
_pdf_reader = LayoutPDFReader(_llm_sherpa_api_url)


@memory.cache
def extract_smart_chunks_from_pdf(pdf_path: Path) -> ExtractedPDFContent:
    doc = _pdf_reader.read_pdf(str(pdf_path))
    chunks: list[Block] = doc.chunks()
    for chunk in chunks:
        print(pdf_path.parent.name, ":", chunk.tag)
        print(chunk.to_context_text(False), "\n\n")
    # TODO: labeled in function of the chunk type
    return { f"Chunk {i} ({chunk.tag})": chunk.to_context_text(False) for i, chunk in enumerate(chunks) }
