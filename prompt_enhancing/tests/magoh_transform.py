import json
from typing import cast
from pathlib import Path
from dspy import Prediction
from archaeo_super_prompt.magoh_target import toMagohData
from archaeo_super_prompt.models.main_pipeline import ExtractedInterventionData
from archaeo_super_prompt.signatures.name import Name
from archaeo_super_prompt.signatures.date_estimation import LatestEstimatedPastMoment

answer = {'source': Prediction(
    reasoning='Il documento riporta una breve annotazione relativa a "Prelievo quote" e "Reperti" in Piazza San Michele a Lucca, con una data specifica. Non si tratta di una relazione estesa, ma sembra essere un verbale o una scheda di registrazione di un\'attività puntuale di prelievo di dati o materiali (quote e reperti). Non sono presenti dettagli che permettano di identificare l\'istituzione responsabile. La tipologia più adatta tra quelle ufficiali è "Scheda dati minimi", in quanto si tratta di una registrazione sintetica di dati relativi a un intervento archeologico.',
    document_source_type='documentazione di indagini archeologiche pregresse',
    institution=None,
    document_type='Scheda dati minimi'
), 'context': Prediction(
    reasoning='Il documento descrive due carotaggi (S1 e S2) effettuati in Piazza San Michele a Lucca, con dettagli stratigrafici e reperti rinvenuti. L\'intervento è chiaramente un "Carotaggio" (sondaggio geognostico), come specificato nel corpo del testo. La data dell\'intervento è indicata come 12/04/23 sia nell\'incipit che all\'inizio delle descrizioni di S1 e S2. Il luogo è la Piazza San Michele, un\'area pubblica centrale di Lucca, e il contesto è quello di indagini stratigrafiche per comprendere la sequenza archeologica e la presenza di strutture antiche (foro romano, attività artigianali medievali, ecc.). Non viene fornito un indirizzo amministrativo preciso oltre al nome della piazza. Il responsabile scientifico (principal investigator) e l\'esecutore materiale non sono esplicitamente menzionati nel testo fornito, quindi non possono essere dedotti. La durata dell\'intervento non è specificata. L\'estensione non è menzionata, quindi va lasciata vuota.',
    municipality='Lucca',
    location='Piazza San Michele',
    address=None,
    place='Piazza pubblica centrale di Lucca, area storica con stratificazioni archeologiche di epoca romana, medievale e moderna, oggetto di carotaggi per indagini stratigrafiche.',
    intervention_date=LatestEstimatedPastMoment(precision='During', date="2023-4-12"),
    intervention_type='Carotaggio',
    duration=None,
    principal_investigator=Name(first_name='', surname=''),
    on_site_qualified_official=[Name(first_name='', surname='')],
    executor='',
    extension=None
), 'technical_achievements': Prediction(
    reasoning='Il rapporto descrive due carotaggi (S1 e S2) effettuati nella Piazza San Michele a Lucca, con stratigrafie dettagliate fino a -4,50 m di profondità. Sono stati recuperati numerosi reperti: frammenti ceramici di varie epoche (romana, tardo-antica, altomedievale, medievale), laterizi, pietre, ossa animali e chiodi di ferro. Sono menzionati anche campioni di pietra, terreno e ghiaia. La stratigrafia mostra una sequenza complessa con livelli di riempimento, accumulo, livelli alluvionali e la presenza di un lastricato romano (blocco di calcare bianco) a circa -3,15/-3,20 m. Non viene indicata la superficie esatta del campo di scavo, né la profondità della falda freatica, anche se si fa riferimento a ristagno d’acqua in alcuni livelli (ma non a una vera e propria falda). La madre roccia non sembra essere stata raggiunta, dato che gli ultimi livelli sono ancora argillosi/alluvionali. L’area è chiaramente pluristratificata, con tracce di frequentazione e attività produttiva (probabile fossa di fusione per campane medievali) e strutture di epoca romana (lastricato del foro).\nPer il conteggio dei campioni: si elencano almeno 2 campioni di pietra (uno per ciascun carotaggio), 1 campione di terreno, 2 campioni di ghiaia arrotondata, e numerosi reperti mobili (ceramica, laterizi, osso, chiodo, ecc.). Considerando i reperti elencati e i campioni specifici, il numero minimo di campioni recuperati è 34 (conteggio dettagliato sotto).',
    sample_number=34,
    field_size=None,
    max_depth=4.5,
    groundwater_depth=None,
    geology=None,
    historical_information_class='sito pluristratificato'
), 'archival_metadata': Prediction(
    reasoning='Il timbro d’archivio riporta la località "Piazza San Michele - Lucca" e la data "12/04/23". Tuttavia, non è presente un numero di protocollo esplicito nel timbro fornito. In assenza di un numero di protocollo identificativo, non è possibile dedurlo o inventarlo. La data di protocollo è chiaramente indicata come "12/04/23".',
    protocol='',
    protocol_date='12/04/23'
)}

with Path(f"./outputs/{"Scheda_Intervento_37172.pdf.ocr.txt"}.prediction.json").resolve().open("w") as json_f:
    json.dump(toMagohData(cast(ExtractedInterventionData, answer)), json_f)
