from typing import Dict, NewType


Chunk = NewType("Chunk", str)
ChunkHumanDescription = NewType("ChunkHumanDescription", str)
Filename = NewType("Filename", str)
ExtractedPDFContent = Dict[ChunkHumanDescription, Chunk]
PDFSources = Dict[Filename, ExtractedPDFContent]
