from pandera.typing.pandas import DataFrame

from ..types.results import ResultSchema


def prettify_field_names(results: DataFrame[ResultSchema]):
    def process_field_name(name: str) -> str:
        splits = name.split("__")
        suffix = splits[0] if len(splits) == 1 else "__".join(splits[1:])
        return suffix

    new_results = results.copy()
    new_results["field_name"] = new_results["field_name"].apply(
        process_field_name
    )
    return new_results
