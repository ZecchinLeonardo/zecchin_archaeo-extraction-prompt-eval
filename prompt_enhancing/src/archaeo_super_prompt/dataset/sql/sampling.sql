select
    *
from
    intervention_data
    TABLESAMPLE SYSTEM (100) REPEATABLE (%(seed)s)
    LIMIT %(max_number)s
