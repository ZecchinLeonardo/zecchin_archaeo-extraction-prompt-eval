from pandera.pandas import DataFrameModel


class NerLabeledChunkDatasetSchema(DataFrameModel):
    """If a chunk is likely to wear information about some data field to be
    extracted, then we add the data field key as a key of the
    nerIdentifiedThesaurus dictionary.
    The best chunks are those in which the list of identified thesaurus is not
    empty for a given identified data field.
    """

    # for each identified field, the list of identified thesaurus
    nerIdentifiedThesaurus: dict[str, list[str]]
