from typing import Iterable, cast
from joblib.memory import MemorizedFunc
from llmsherpa.readers import Block, LayoutPDFReader
from pathlib import Path


from ..debug_log import print_warning
from ..types.intervention_id import InterventionId
from ..types.pdfchunks import (
    PDFChunk,
    PDFChunkDataset,
    PDFChunkDatasetSchema,
    buildPdfChunkDataset,
    composePdfChunkDataset,
)

from ..cache import get_memory_for
from ..signatures.input import Chunk, Filename

# TODO: edit it with .env
_llm_sherpa_api_url = (
    "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=yes"
)
_pdf_reader = LayoutPDFReader(_llm_sherpa_api_url)


def _extract_smart_chunks_from_pdf(
    filepath: str, intervention_id: int
) -> PDFChunkDataset:
    """Type-unsafe intern function for cache feature"""
    filename = Filename(Path(filepath).name)
    chunks: list[Block] = []
    try:
        # llmsherpa has unhandled KeyErrors ('return_dict')
        # in json loading of the parsed values
        # so it might happen on unprocessable pdfs
        doc = _pdf_reader.read_pdf(filepath)
        chunks = doc.chunks()
    except* Exception as e:
        print_warning(
            f"Getting the chunks or the read pdf has not worked for id {intervention_id} with file {filename}:\n{str(e)}",
        )
    if not chunks:
        return PDFChunkDataset(PDFChunkDatasetSchema.empty())
    total_page_number = max(chunk.page_idx for chunk in chunks)

    def get_row(chunk: Block, chunk_nb: int):
        try:
            chunk_ctx_content = Chunk(chunk.to_context_text())
            chunk_page_position = f"{chunk.page_idx}/{total_page_number}"
            return PDFChunk(
                {
                    "id": InterventionId(intervention_id),
                    "filename": filename,
                    "chunk_index": chunk_nb,
                    "chunk_content": chunk_ctx_content,
                    "chunk_page_position": chunk_page_position,
                    "chunk_type": chunk.tag,
                }
            )
        except* Exception as e:
            print_warning(
                f"Getting the chunks has not worked for id {intervention_id} with file {filename}:\n{str(e)}"
            )
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


_cached_func = cast(MemorizedFunc, get_memory_for("interim").cache(_extract_smart_chunks_from_pdf))

class UnreadableSourceSetError(Exception):
    pass

def extract_smart_chunks_from_pdfs_of_intervention(
    pdf_paths: Iterable[str], intervention_id: InterventionId
) -> PDFChunkDataset:
    results = [
        _cached_func.call_and_shelve(pdf_path, int(intervention_id))
        for pdf_path in pdf_paths
    ]
    chunkDataset = composePdfChunkDataset((result.get() for result in results))
    if chunkDataset.empty:
        # clear the invalid cache
        for result in results:
            result.clear()
        raise UnreadableSourceSetError
    return chunkDataset
    
