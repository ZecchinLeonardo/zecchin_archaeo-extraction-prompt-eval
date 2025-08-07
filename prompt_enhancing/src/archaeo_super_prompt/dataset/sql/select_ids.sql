select
    *
from
    featured__intervention_data
where "scheda_intervento.id" in %(intervention_ids)s;
