from typing import NewType


Chunk = NewType("Chunk", str)
ChunkHumanDescription = NewType("ChunkHumanDescription", str)
Filename = NewType("Filename", str)
ExtractedPDFContent = dict[ChunkHumanDescription, Chunk]
PDFChunkEnumeration = str
"""
The enumeration is for now done with a self-engineered mardkown rendering

But, we might consider giving to the llm another kind of structure, like a json 
TODO: read documentation about that and transform if needed in the optimization iterations
"""

PDFSources = dict[Filename, ExtractedPDFContent]
