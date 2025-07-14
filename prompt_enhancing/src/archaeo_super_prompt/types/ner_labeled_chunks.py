from typing import Dict, List
from pandera import DataFrameModel

class NerLabeledChunkDatasetSchema(DataFrameModel):
    # for each identified field, the list of identified thesaurus
    nerIdentifiedThesaurus: Dict[str, List[str]]
