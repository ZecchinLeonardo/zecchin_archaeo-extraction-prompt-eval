from typing import List, Optional
import dspy

"""In the code we declare both semantic and typing data for dspy to augment
prompt efficiency to get the expected output from the LLM """
class ExtractionArcheoData(dspy.Signature):
    """Extract structured information from text."""

    italian_archaeological_document: str = dspy.InputField(desc="This is an Italian legal document reporting an archaeological intervention in some place")

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
    # is this the date of excavation ? or is it an administraive date ?
    # TODO: play for having better types (such as, date periods)
    intervention_date: List[str] = dspy.OutputField(desc="List of periods during which the intervention was carried out.")
    # TODO: help the model on intervention type
    intervention_type: str = dspy.OutputField(desc="This describe the archaeological intervention type")
    duration: Optional[int] = dspy.OutputField(desc="The duration in number of days")
    done_since: str = dspy.OutputField(desc="Date since the end of the intervention")
    # generally , scientific direction and qualified people are the samee
    # all the names are in this format: N. Cognome
    # in documents, there is always full name (nome e cognome)
    scientific_direction: str = dspy.OutputField(
        desc="Name/Title of the supervisor and/or the supervising institution"
    )
    extension: List[str] = dspy.OutputField(
        desc="If more than one intervention, list them here."
    )
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
    # TODO: ask the title (Dott., Arch., etc.)
    # one people or a list
    on_site_qualified_official: List[str] = dspy.OutputField(desc="List of the archaeologists beginning by their abbreviated title following their name")
    # TODO: help with list of values
    # on normal pdf, usually in the title or in the first sentence 
    document_type: str = dspy.OutputField(desc="A label for classifying this archaeological document")
    
    # these two fields vary along the archival system and the time when it has
    # been archieved
    protocol: Optional[str] = dspy.OutputField(desc="Identifier of the protocol, if there is one") 
    protocol_date: Optional[str] = dspy.OutputField(desc="The date of the identified protocol bound to this intervention.")
