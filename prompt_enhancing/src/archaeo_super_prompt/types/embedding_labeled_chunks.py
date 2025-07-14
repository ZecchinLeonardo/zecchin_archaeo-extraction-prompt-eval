from typing import Dict, List

from pandera import DataFrameModel

class SemanticallyLabeledChunkDatasetSchema(DataFrameModel):
    # for each identified field, the list of identified thesaurus
    semanticallyIdentifiedThesaurus: Dict[str, List[str]]
