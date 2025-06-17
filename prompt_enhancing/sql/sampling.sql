select
    *
from
    intervention_data
    TABLESAMPLE SYSTEM (100) REPEATABLE (:seed)
    LIMIT :max_number
