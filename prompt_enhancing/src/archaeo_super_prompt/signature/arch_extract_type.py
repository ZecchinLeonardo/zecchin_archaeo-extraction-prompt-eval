"""The field descriptions are samples or interpretations of this paper:
MapPapers 1en-II, 2012, pp.21-38
doi:10.4456/MAPPA.2012.02
"""

from typing import List, Optional, Union
import dspy

from .date_estimation import LatestEstimatedPastMoment
from .document_type import ItalianDocumentType
from .intervention_type import ItalianInterventionType
from .name import Name
from .ogd import ItalianOGD
from .ogm import ItalianOGM

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

    # --------------------------- INPUTS ----------------------------

    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    # The document will be in the format of a text extracted from an OCR operation in a PDF, then 
    italian_archaeological_document: str = dspy.InputField(desc="""This is an Italian offical archive document reporting an archaeological intervention in some Italian place.
    It is the output of an OCR that has been applied on a PDF file which is generally in a numerical clean format. But, sometimes the original document is a scan of a paper (I let you work with this information unknown). The text is therefore divided into blocks, each carrying different classes of information. The overall set of classes of information is nonetheless the same and enables to identify most of the data that will be requested. This is just the order of the blocks that can differ over the documents in cause of the OCR operation.
    Please note that among these blocks, there is a small one with an artificial format which represents a stamp. This stamp will allow to directly identify some precise fields.
    """)

    # --------------------------- INPUTS ----------------------------

    # --------------------------- OUTPUTS ---------------------------

    # ------ Topographic and technical data ------

    # TODO: ask more precisions about the positional fields
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
    intervention_date: LatestEstimatedPastMoment = dspy.OutputField(date="Moment of the intervention with at least the year when it started. You can also precise the month or even precise date if enough information is provided. The document may not mention the interventin date, therefore, if it is the case, you have to answer that this is before the date of archiving you have figured out (then, you must precise this date)")
    intervention_type: ItalianInterventionType = dspy.OutputField()
    duration: Optional[int] = dspy.OutputField(desc="The duration of the intervention, expressed in working days")

    # generally, scientific direction and qualified people are the same
    # in documents, there is always full name (nome e cognome)
    principal_investigator: Name = dspy.OutputField(
        desc="Name of the scientific supervisor of the intervention. This is generally the qualified professional arcaheologist or academician that has worked on site."
    )
    # one people or a list: generally one people
    on_site_qualified_official: List[Name] = dspy.OutputField(desc="List of the archaeologists on site. Generally, there is just one person and this is the principal investigator")
    executor: Union[Name, str] = dspy.OutputField(desc="""The name of the person, team, company or institution who/which materially performed the intervention.""")

    extension: List[str] = dspy.OutputField(
        desc="If there are more than one excavation, list them here. This information is generally empty"
    )

    sample_number: int = dspy.OutputField(desc="The number of samples recovered on site during this intervention")
    field_size: Optional[float] = dspy.OutputField(desc="The sample area of the excavation in square metres.")
    max_depth: float = dspy.OutputField(desc="The absolute value of the maximum depth (in metres) reached during the excavation")
    # this field is not filled most of the time, while this is though useful information
    # if water is reacht, then generally profondita_equal ~= max_depth
    groundwater_depth: Optional[float] = dspy.OutputField(desc="""The absolute value of the depth (in metres) at which groundwater was met. This value is very approximate. Documents reveal that the point of groundwater surfacing is not systematically calculated. Since probably not considered an important issue, it is available only when water surfacing compromises activities, making stratigraphic readings difficult or forcing excavations to be suspended. There are only few documents that report this value and, of these, many are approximate""")
    # if we have reacht the mother rock (make it optional, because sometimes it
    # is not found)
    geology: Optional[bool] = dspy.OutputField(desc="If you have this information, assess if the mother rock has been reached and then if all the possible historical eras have been inspected")

    # TODO:
    purpose_of_ogd_draw: ItalianOGD = dspy.OutputField(desc="")

    # ------ Topographic and technical data ------

    # ------ Chronological data ------

    # TODO:

    # ------ Chronological data ------
    
    # ------ Source of Information data ------

    # TODO:
    purpose_of_ogm_museum: ItalianOGM = dspy.OutputField(desc="")
    institution: Optional[str] = dspy.OutputField(desc="Italian title of the Archaeology institution carrying out the intervention")
    document_type: ItalianDocumentType = dspy.OutputField(desc="A label for classifying this archaeological document. If you can recognize the title or the first sentence of the document body, then you shall identify this type there.")
    # these two fields vary along the archival system and the time when it has
    # been archieved
    # TODO:
    protocol: Optional[str] = dspy.OutputField(desc="Identifier of the protocol, noted on the stamp")
    protocol_date: Optional[str] = dspy.OutputField(desc="The date of the archival in the protocol, noted on the stamp")

    # ------ Source of Information data ------

    # --------------------------- OUTPUTS ---------------------------
