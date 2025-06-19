select
    *
from
    intervention_data
    ORDER BY RANDOM()
    LIMIT %(max_number)s
where
    TO_DATE("building.Data Protcollo", 'DD-MM-YYYY') >= TO_DATE('01-01-2015', 'DD-MM-YYYY')
