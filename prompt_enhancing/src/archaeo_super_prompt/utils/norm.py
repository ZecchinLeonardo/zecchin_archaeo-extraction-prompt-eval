from functools import reduce
from typing import cast


def variabilize_column_name(stringified_column_name: str) -> str:
    return stringified_column_name.replace(" ", "_").replace(".", "__")


def flatten_dict[T](d: dict[str, dict[str, T]]) -> dict[str, T]:
    return reduce(
        lambda flat_dict, first_depth_item: flat_dict
        | {
            variabilize_column_name(f"{first_depth_item[0]}.{second_k}"): v
            for second_k, v in cast(dict, first_depth_item[1]).items()
        },
        d.items(),
        {},
    )
