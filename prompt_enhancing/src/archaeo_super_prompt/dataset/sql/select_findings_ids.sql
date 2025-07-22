select findings.* from samples inner join findings
on samples."scheda_intervento.id" = findings."scheda_intervento.id"
where "scheda_intervento.id" in (%(intervention_ids)s);
