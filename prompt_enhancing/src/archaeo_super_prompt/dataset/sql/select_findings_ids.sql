select
    *
from
    findings
    WHERE "scheda_intervento.id" in %(intervention_ids)s;
