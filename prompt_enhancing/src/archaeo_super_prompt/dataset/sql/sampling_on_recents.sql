select *
from
    featured__intervention_data
where
    TO_DATE("building.Data Protocollo", 'DD-MM-YYYY')
    >= TO_DATE('01-01-2015', 'DD-MM-YYYY')
order by RANDOM()
limit %(max_number)s
