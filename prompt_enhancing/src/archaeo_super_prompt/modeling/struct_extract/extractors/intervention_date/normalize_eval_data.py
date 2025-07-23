"""We set a dspy model to adapt the training intervention date data with our specification."""

import dspy
from scipy.sparse import data

from .type_models import Data, Precisione


class StimareFinestraDiData(dspy.Signature):
    """Della descrizione della data di partenza di un'archiviata indagine, trova una finestra di due date, con un precisione al giorno, al mese o all'anno più vicino. Quando la descrizione dice solamente "pre", "prima di", "dopo", etc., si rifere a la data di archiviazone.

    1. Innanzitutto, determina la precisione con cui puoi approssimare la finestra.
    2. Quindi, determina la finestra, inserendo valori predefiniti (ma ben tipizzati) nei campi non coperti dalla precisione.
       a. Se e rilevante, restringi la finestra a un punto impostando le stesse date minima e massima.
    """

    descrizione_di_inizio: str = dspy.InputField()
    data_di_archiviazone: Data = dspy.InputField()
    data_minima_di_inizio: Data = dspy.OutputField()
    data_massima_di_inizio: Data = dspy.OutputField()
    precisione: Precisione = dspy.OutputField()


class EstimateInterventionDate(dspy.Module):
    """Dspy model for inferring a window of dates."""

    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self._estrattore_delle_date = dspy.ChainOfThought(
            StimareFinestraDiData
        )

    def forward(self, descrizione_di_inizio: str, data_di_archiviazone: str):
        return self._estrattore_delle_date(
            descrizione_di_inizio=descrizione_di_inizio,
            data_di_archiviazone=data_di_archiviazone,
        )
