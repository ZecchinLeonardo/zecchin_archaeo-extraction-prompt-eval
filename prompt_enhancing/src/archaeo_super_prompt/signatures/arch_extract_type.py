"""The field descriptions are samples or interpretations of this paper:
MapPapers 1en-II, 2012, pp.21-38
doi:10.4456/MAPPA.2012.02
"""

from typing import Dict, List, Optional, Union
import dspy

from ..signatures.input import ExtractedPDFContent
from .date_estimation import LatestEstimatedPastMoment
from .document_type import ItalianDocumentType
from .intervention_type import ItalianInterventionType
from .name import Name
from .ogd import FinalItalianOGD
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

# TODO: give examples in descriptions
class ArchaeologicalReportCutting(dspy.Signature):
    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    # The document will be in the format of a text extracted from an OCR operation in a PDF, then 
    italian_document_ocr_scan: Dict[str, ExtractedPDFContent] = dspy.InputField(desc="""This is a set of Italian offical archive documents reporting an archaeological intervention in some Italian place. The key is the name of the source and the value is its content, divided into layout-labeled chunks. There is always one document which contains the main report of the intervention. From it you can extract most of the information.
    The content is extracted from the output of an OCR that has been applied on a PDF file which is generally in a numerical clean format. But, sometimes the original document is a scan of a paper (even hand-writtent papers) (I let you work with this information unknown).
    Please note that among these chunks, there is a small block of text with an artificial format which represents a stamp. This stamp will allow to directly identify some precise fields.
    """)

    incipit: str = dspy.OutputField(desc="The header and first content part of the main report document, with all its metadata.")
    archival_stamp: Optional[str] = dspy.OutputField(desc="A small part of non-natural text with a protocol number and a date, findable in the main report document.")
    body: dict[str, ExtractedPDFContent] = dspy.OutputField(desc="To each document, the remaining content, divided into chunks as in the input.")

class ArchaeologicalInterventionContext(dspy.Signature):
    """Extract structured information about an archaeological intervention from an official archive document."""

    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    archaeological_report_incipit: str = dspy.InputField(desc="The header and first content part of the archaeological intervention report, with all its metadata.")
    archaeological_report_body: str = dspy.InputField(desc="All the content of the report concretely describing the archaeological intervention.")

    # TODO: add the thesaurii
    municipality: str = dspy.OutputField(
        desc="(mandatory) Italian territorial entity where the archaeological intervention took place."
    )
    # TODO: add the thesaurii
    location: str = dspy.OutputField(desc="(mandatory) Title for identifying the place (institution, parking, farm property, etc.) where the archaeological intervention took place."
    )
    # TODO: add the thesaurii
    address: Optional[str] = dspy.OutputField(
        desc="Precise administrative address (street, number) of the intervention's place."
    )
    place: str = dspy.OutputField(
        desc="(mandatory) More informal/natural description of the place and what there was there at the moment of the intervention."
    )
    intervention_date: Optional[Union[LatestEstimatedPastMoment, str]] = dspy.OutputField(date="(mandatory) Date of the intervention with at least the year when it started. You can also precise the month or even precise date if enough information is provided. The document may not mention the intervention date, therefore, if it is the case, you have to answer that this is before the date of archiving you have figured out (then, you must precise this date)")
    intervention_type: ItalianInterventionType = dspy.OutputField(desc="(mandatory) You do not have to invent it. It must be among the given set of values (they are official types of intervention in the academical institutions)")
    duration: Optional[int] = dspy.OutputField(desc="The duration of the intervention, expressed in working days")

    # generally, scientific direction and qualified people are the same
    # in documents, there is always full name (nome e cognome)
    # TODO: remove the Union and force the name
    principal_investigator: Optional[Union[Name, str]] = dspy.OutputField(
        desc="(mandatory) Name of the scientific supervisor of the intervention. This is the qualified professional arcaheologist or academician that has worked on site. His or her name is always written somewhere so you must find him/her."
    )
    # one people or a list: generally one people
    on_site_qualified_official: List[Union[Name, str]] = dspy.OutputField(desc="(at least the prinicpal inverstigator in the list) List of the archaeologists on site. Generally, there is just one person and this is the principal investigator")
    executor: Optional[Union[Name, str]] = dspy.OutputField(desc="""(mandatory) The name of the person, team, company or institution who/which materially performed the intervention.""")

    extension: Optional[List[str]] = dspy.OutputField(
        desc="If there are more than one excavation, list them here. This information is generally empty"
    )

class TechnicalInformation(dspy.Signature):
    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    archaeological_report_body: str = dspy.InputField(desc="All the content of the report concretely describing the archaeological intervention. You should use it in priority for extracting archaeological technical data.")

    sample_number: int = dspy.OutputField(desc="(mandatory) The number of samples recovered on site during this intervention")
    field_size: Optional[float] = dspy.OutputField(desc="The sample area of the excavation in square metres.")
    max_depth: Optional[float] = dspy.OutputField(desc="The absolute value of the maximum depth (in metres) reached during the excavation")
    # this field is not filled most of the time, while this is though useful information
    # if water is reacht, then generally profondita_equal ~= max_depth
    groundwater_depth: Optional[float] = dspy.OutputField(desc="""The absolute value of the depth (in metres) at which groundwater was met. This value is very approximate. Documents reveal that the point of groundwater surfacing is not systematically calculated. Since probably not considered an important issue, it is available only when water surfacing compromises activities, making stratigraphic readings difficult or forcing excavations to be suspended. There are only few documents that report this value and, of these, many are approximate""")
    # if we have reacht the mother rock (make it optional, because sometimes it
    # is not found)
    geology: Optional[bool] = dspy.OutputField(desc="If you have this information, assess if the mother rock has been reached and then if all the possible historical eras have been inspected")
    # TODO: change it with a List of findings
    historical_information_class: FinalItalianOGD = dspy.OutputField(desc="(mandatory) It is a label among a normalized set of classes to qualify the historical information that the objects found during the intervention bring for a place and an epocha. Please notice that there are the special labels for an intervention when nothing has been found, when the information is unkown and (the very most common situation) when several classes occur in this intervention.")


# ------ Chronological data ------

# TODO:

# ------ Chronological data ------

class SourceOfInformationInReport(dspy.Signature):
    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    report_incipit: str = dspy.InputField()

    document_source_type: ItalianOGM = dspy.OutputField(desc="(mandatory) A class of document source among a set of official values to identify which kind of institution has processed the current report.")
    institution: Optional[str] = dspy.OutputField(desc="Italian title of the Archaeology institution carrying out the intervention. It is directly written in the report then you do not have to invent it.")
    document_type: ItalianDocumentType = dspy.OutputField(desc="(mandatory) An italian label for classifying this archaeological document. If you can recognize the title or the first sentence of the document body, then you shall identify this type there. It must be among the label set that I gave you (these are official labels).")

class ArchivalInformation(dspy.Signature):
    context: str = dspy.InputField(desc="Facts here are assumed to be true and must be taken in account for the analysis to suitably process")
    report_archive_office_stamp: str = dspy.InputField()

    # these two fields vary along the archival system and the time when it has
    # been archieved
    # TODO:
    protocol: str = dspy.OutputField(desc="Identifier of the protocol, with a range of digits and sometimes a hyphen with a letter as suffix")
    protocol_date: str = dspy.OutputField(desc="The date of the archival in the protocol")
