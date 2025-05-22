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
    scientific_direction: str = dspy.OutputField(
        desc="Name/Title of the supervisor and/or the supervising institution"
    )
    extension: List[str] = dspy.OutputField(
        desc="If more than one intervention, list them here."
    )
    test_number: int = dspy.OutputField(desc="Index of the archaeological trial ")
    # TODO: ask
    max_depth: float = dspy.OutputField(desc="The maximum depth of excavation that occured during the intervention")
    geology: str = dspy.OutputField(desc="Field geological description")
    # TODO: ask
    diD_stuff: List[str] = dspy.OutputField(desc="List the diD objects that were used during the intervention, if there is")
    # TODO: ask
    ogm_museum_stuff: List[str] = dspy.OutputField(desc="List the stuff used from the OGM museum during the intervention, if there is.")
    # TODO: ask for correct translation
    falda_depth: Optional[str] = dspy.OutputField(desc="Description of the depth of falda")
    institution: Optional[str] = dspy.OutputField(desc="Italian title of the Archaeology institution carying out the intervention")
    # TODO: ask the title (Dott., Arch., etc.)
    on_site_qualified_official: List[str] = dspy.OutputField(desc="List of the archaeologists beginning by their abbreviated title following their name")
    # TODO: help with list of values
    document_type: str = dspy.OutputField(desc="A label for classifying this archaeological document")
    protocol: Optional[str] = dspy.OutputField(desc="Identifier of the protocol, if there is one") 
    protocol_date: Optional[str] = dspy.OutputField(desc="The date of the identified protocol bound to this intervention.")
