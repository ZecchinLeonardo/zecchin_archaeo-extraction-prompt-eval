select
    *
from
    featured__intervention_data
    ORDER BY RANDOM()
    LIMIT %(max_number)s
