
from typing import List, Optional
import dspy

from .date_estimation import LatestEstimatedPastMoment
from .document_type import ItalianDocumentType
from .intervention_type import ItalianInterventionType
from .name import Name

"""In the code, we declare both semantic and typing data for dspy to augment
prompt efficiency to get the expected output from the LLM. Those are the
code elements that are used to compute the prompt:
- attributes and class names
- attributes and class docstrings
- Field description attributes
- attributes typing

NB: almost everything is used
"""

class ArchaeologicalInterventionData(dspy.Signature):
    """Extract structured information about an archaeological intervention from an official archive document."""

    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    # The document will be in the format of a text extracted from an OCR operation in a PDF, then 
    italian_archaeological_document: str = dspy.InputField(desc="""This is an Italian archaeological archive document reporting an archaeological intervention in some Italian place.
    It is the output of an OCR that has been applied on a PDF file which is generally in a numerical clean format. But, sometimes the original document is a scan of a paper (I let you work without this information). The text is therefore divided into blocks, each carrying different classes of information. The overall set of classes of information is nonetheless the same and enables to identify most of the data that will be requested. This is just the order of the blocks that can differ over the documents in cause of the OCR operation.""")

    municipality: str = dspy.OutputField(
        desc="Italian territorial entity where the archaeological intervention took place."
    )
    location: str = dspy.OutputField(
        desc="Title for identifying the place (institution, parking, farm property, etc.) where the archeological intervention took place."
    )
    address: Optional[str] = dspy.OutputField(
        desc="Precise administrative address (street, number) of the intervention's place."
    )
    place: str = dspy.OutputField(
        desc="Non-administrative description of the address"
    )
    intervention_date: LatestEstimatedPastMoment = dspy.OutputField()
    intervention_type: ItalianInterventionType = dspy.OutputField()
    duration: Optional[int] = dspy.OutputField(desc="The duration in number of days")
    done_since: LatestEstimatedPastMoment = dspy.OutputField(desc="Date since the end of the intervention. If you lack information, then it will generally be in the same time as the intervention date")
    # generally , scientific direction and qualified people are the same
    # in documents, there is always full name (nome e cognome)
    scientific_direction: Name = dspy.OutputField(
        desc="Name of the supervisor of the intervention. This is generally the qualified professional arcaheologist or academician that has worked on site"
    )
    extension: List[str] = dspy.OutputField(
        desc="If there are more than one intervention, list them here. This information is generally empty"
    )
    # TODO:
    test_number: int = dspy.OutputField(desc="Index of the archaeological trial ")
    # TODO: ask
    # "-$float" in metres, and the point is the .
    max_depth: float = dspy.OutputField(desc="The maximum depth reacht during the excavation")
    # if we have reacht the mother rock (make it optional, because sometimes it
    # is not found)
    geology: str = dspy.OutputField(desc="Field geological description")
    # TODO: ask
    diD_stuff: List[str] = dspy.OutputField(desc="List the diD objects that were used during the intervention, if there is")
    # TODO: ask
    ogm_museum_stuff: List[str] = dspy.OutputField(desc="List the stuff used from the OGM museum during the intervention, if there is.")
    # this field is not field most of the time, while this is though useful
    # informatoin
    # say if we have reacht a depth where water is gonna be present 
    # if water is reacht, then generally profondita_equal ~= max_depth
    falda_depth: Optional[str] = dspy.OutputField(desc="Description of the depth of falda")

    institution: Optional[str] = dspy.OutputField(desc="Italian title of the Archaeology institution carying out the intervention")
    # one people or a list: generally one people
    on_site_qualified_official: List[str] = dspy.OutputField(desc="List of the archaeologists beginning by their abbreviated title following their name")
    document_type: ItalianDocumentType = dspy.OutputField(desc="A label for classifying this archaeological document. If you can recognize the title or the first sentence of the document body, then you shall identify this type.")
    
    # these two fields vary along the archival system and the time when it has
    # been archieved
    protocol: Optional[str] = dspy.OutputField(desc="Identifier of the protocol, if there is one") 
    protocol_date: Optional[str] = dspy.OutputField(desc="The date of the identified protocol bound to this intervention.")
