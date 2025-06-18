from typing import Dict, List


Chunks = List[str]
Filename = str
ExtractedPDFContent = Dict[str, Chunks]
PDFSources = Dict[Filename, ExtractedPDFContent]
