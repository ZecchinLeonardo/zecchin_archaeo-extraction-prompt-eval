from .pdfchunks import PDFChunkDatasetSchema
from .embedding_labeled_chunks import SemanticallyLabeledChunkDatasetSchema
from .ner_labeled_chunks import NerLabeledChunkDatasetSchema


class FeaturedChunks(PDFChunkDatasetSchema, NerLabeledChunkDatasetSchema,
                    SemanticallyLabeledChunkDatasetSchema):
    """A Dataframe of chunks with hints for some fields.

    These hints are those pre-extracted from the NER model and the
    EmbeddingModel
    """
    pass
