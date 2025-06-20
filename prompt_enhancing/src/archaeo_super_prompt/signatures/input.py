from typing import Dict


Chunks = str
Filename = str
ExtractedPDFContent = Dict[str, Chunks]
PDFSources = Dict[Filename, ExtractedPDFContent]
