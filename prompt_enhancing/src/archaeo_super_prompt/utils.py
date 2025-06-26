from functools import reduce
from typing import Dict, cast


def flatten_dict[T](d: Dict[str, Dict[str, T]]) -> Dict[str, T]:
    return reduce(
        lambda flat_dict, first_depth_item: flat_dict
        | {
            f"{first_depth_item[0]}.{second_k}": v
            for second_k, v in cast(dict, first_depth_item[1]).items()
        },
        d.items(),
        {},
    )

