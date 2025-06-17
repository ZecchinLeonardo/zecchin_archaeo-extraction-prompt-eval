with samples as (
-- sampling-placeholder
)

select findings.* from samples inner join findings
on samples."scheda_intervento.id" = findings."scheda_intervento.id";
