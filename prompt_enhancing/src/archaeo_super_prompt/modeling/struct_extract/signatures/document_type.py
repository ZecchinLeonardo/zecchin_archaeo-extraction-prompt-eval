from typing import Literal
from bidict import bidict, BidirectionalMapping

DocumentType = Literal[
    "Assignment Reports",
    "Excavation Reports",
    "State of Advancement Reports",
    "Information Reports/Notices",
    "Collection Reports",
    "Requests for Authorisation",
    "Communications",
    "Photographic images",
    "Basic data form",
    "Other",
    "Investigation report",
    "Plans/Drawings",
]

ItalianDocumentType = Literal[
    "Relazione di missione",
    "Relazione di scavo",
    "Relazione stato lavori",
    "Informativa/segnalazione",
    "Verbale di ritiro",
    "Richiesta autorizzazione",
    "Comunicazione",
    "Immagini fotografiche",
    "Scheda dati minimi",
    "Altro",
    "Relazione indagine",
    "Piante/Disegni",
]

TO_ITALIAN_DOCUMENT_TYPE: BidirectionalMapping[DocumentType, ItalianDocumentType] = (
    bidict(
        [
            ("Assignment Reports", "Relazione di missione"),
            ("Excavation Reports", "Relazione di scavo"),
            ("State of Advancement Reports", "Relazione stato lavori"),
            ("Information Reports/Notices", "Informativa/segnalazione"),
            ("Collection Reports", "Verbale di ritiro"),
            ("Requests for Authorisation", "Richiesta autorizzazione"),
            ("Communications", "Comunicazione"),
            ("Photographic images", "Immagini fotografiche"),
            ("Basic data form", "Scheda dati minimi"),
            ("Other", "Altro"),
            ("Investigation report", "Relazione indagine"),
            ("Plans/Drawings", "Piante/Disegni"),
        ]
    )
)
