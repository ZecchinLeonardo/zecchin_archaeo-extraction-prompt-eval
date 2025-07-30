from typing import Literal, Union

"""
In most of the case, the sample in the training dataset will have the value
"sito pluristradificato"

"""

ItalianOGD = Literal[
    "area a uso funerario",
    "area di materiale mobile",
    "elemento toponomastico",
    "giacimento in cavit naturale",
    "giacimento paleontologico",
    "giacimento subacqueo",
    "infrastruttura agraria",
    "infrastruttura assistenziale",
    "infrastruttura di consolidamento",
    "infrastruttura di servizio",
    "infrastruttura idrica",
    "infrastruttura portuale",
    "infrastruttura viaria",
    "insediamento",
    "luogo ad uso pubblico",
    "luogo commemorativo",
    "luogo con deposizione di materiale",
    "luogo con elemento per la confinazione",
    "luogo con ritrovamento sporadico",
    "luogo con tracce di frequentazione",
    "luogo di attivit produttiva",
    "struttura abitativa",
    "struttura di fortificazione",
    "strutture per il culto",
]

SpecialItalianOGD = Literal[
    "sito pluristratificato",
    "area priva di tracce archeologiche",
    "sito non identificato",
]

FinalItalianOGD = Union[ItalianOGD, SpecialItalianOGD]

"""The final italian ogd can be figured out with this algorithm
"""

# TODO: define this in function of the thesaura
# the AI will first have to find out a list of these things
Finding = Literal[""]


def get_ogd_of_finding(finding: Finding) -> ItalianOGD:
    # TODO:
    finding = finding
    return "area a uso funerario"


def process_with_reading_document() -> SpecialItalianOGD:
    # TODO:
    return "sito non identificato"


def get_ogd_from_finding_list(findings: list[Finding]) -> FinalItalianOGD:
    if len(findings) > 1:
        return "sito pluristratificato"
    if len(findings) == 0:  # unusual situation in reading reports
        return process_with_reading_document()
    return get_ogd_of_finding(findings[0])
