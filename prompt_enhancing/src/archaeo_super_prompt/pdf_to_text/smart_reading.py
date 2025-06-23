from llmsherpa.readers import Block, LayoutPDFReader
from pathlib import Path


from archaeo_super_prompt.types.intervention_id import InterventionId
from archaeo_super_prompt.types.pdfchunks import (
    PDFChunk,
    PDFChunkDataset,
    buildPdfChunkDataset,
)

from ..cache import memory
from ..signatures.input import Chunk, Filename

# TODO: edit it with .env
_llm_sherpa_api_url = (
    "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=yes"
)
_pdf_reader = LayoutPDFReader(_llm_sherpa_api_url)


@memory.cache
def extract_smart_chunks_from_pdf(
    pdf_path: Path, intervention_id: InterventionId
) -> PDFChunkDataset:
    filename = Filename(pdf_path.name)
    doc = _pdf_reader.read_pdf(str(pdf_path))
    chunks: list[Block] = doc.chunks()
    if not chunks:
        raise Exception("No content to be extracted")
    total_page_number = max(chunk.page_idx for chunk in chunks)

    def get_row(chunk: Block, chunk_nb: int):
        try:
            chunk_ctx_content = Chunk(chunk.to_context_text())
            chunk_page_position = f"{chunk.page_idx}/{total_page_number}"
            return PDFChunk(
                {
                    "id": intervention_id,
                    "filename": filename,
                    "chunk_index": chunk_nb,
                    "chunk_content": chunk_ctx_content,
                    "chunk_page_position": chunk_page_position,
                    "chunk_type": chunk.tag,
                }
            )
        except* Exception:
            pass
        return None

    return buildPdfChunkDataset(
        [
            opt_row
            for opt_row in [
                get_row(chunk, chunk_idx) for chunk_idx, chunk in enumerate(chunks)
            ]
            if opt_row is not None
        ]
    )
