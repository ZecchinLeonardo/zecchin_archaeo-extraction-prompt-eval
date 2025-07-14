from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, cast
from ....types.structured_data import (
    ExtractedStructuredDataSeries,
    outputStructuredDataSchema,
)
from ....utils.norm import flatten_dict
from dspy import Example, Prediction
import dspy
from dspy.evaluate.metrics import answer_exact_match

from .similarity_match import soft_accuracy

from .smart_match_checking import check_date_with_LLM


MIN_VALUE = 0
MAX_VALUE = 1

U, V = TypeVar("U"), TypeVar("V")


def _validate_magoh_data(
    answer: ExtractedStructuredDataSeries,
    pred: ExtractedStructuredDataSeries,
    trace=None,
) -> dict[str, float] | dict[str, bool]:
    def check_null[U, V](func: Callable[[U, V], bool]):
        def inner(e: Union[U, None], p: Union[V, None]):
            if (e is None) == (p is None):
                if e is None:
                    return True
                return func(e, cast(V, p))
            return False

        return inner

    def validate_type[U](func: Callable[[U, U], bool]):
        @check_null
        def inner(e: U, p: Any):
            if isinstance(p, type(e)):
                return func(p, e)
            return False

        return inner

    def iterate_if_needed[U](func: Callable[[U, U], bool]):
        @validate_type
        def inner(
            e: Union[U, List[U], Tuple[U]], p: Union[U, List[U], Tuple[U]]
        ):
            if isinstance(e, List) or isinstance(e, Tuple):
                print(f"{e=}")
                e_list = cast(Union[List[U], Tuple[U]], e)
                p_list = cast(Union[List[U], Tuple[U]], p)
                if len(p_list) != len(e_list):
                    return False
                return all(inner(e_list[i], p_list[i]) for i in range(len(e)))
            else:
                return func(e, cast(U, p))

        return inner

    @iterate_if_needed
    def perfect_match[U](e: U, p: U):
        # if e and p perfectly match with other types than str,
        # then their stringified version will perfectly match
        return answer_exact_match(
            dspy.Example(answer=str(e)), dspy.Prediction(answer=str(p)), trace
        )

    @validate_type
    def complex_match(e: str, p: str):
        TRESHOLD = 0.75
        # llm_check = check_with_LLM(TRESHOLD)
        # return llm_check(e, p, trace) == 1
        similarity_check = soft_accuracy([e], [p], TRESHOLD)["matches"].item()
        return similarity_check

    # unused
    @validate_type
    def neutral(e: str, p: str):
        e = e
        p = p
        return False

    neutral = neutral

    @validate_type
    def date_compare(e: str, p: str):
        # TODO: avoid LLM when the typing will be better
        return check_date_with_LLM(e, p, trace) == 1

    pred_to_compare = pred
    metrics: Dict[str, Dict[str, Callable[[Any, Any], bool]]] = {
        "university": {
            "Sigla": perfect_match,  # TODO: figure this out
            "Comune": complex_match,
            "Ubicazione": complex_match,
            "Indirizzo": complex_match,
            "Località": complex_match,
            "Data intervento": date_compare,
            "Tipo di intervento": perfect_match,
            "Durata": perfect_match,
            "Eseguito da": perfect_match,
            "Direzione scientifica": perfect_match,
            "Estensione": complex_match,
            "Numero di saggi": perfect_match,
            "Profondità massima": perfect_match,
            "Geologico": perfect_match,
            "OGD": perfect_match,
            "OGM": perfect_match,
            "Profondità falda": perfect_match,
        },
        "building": {
            "Istituzione": complex_match,
            "Funzionario competente": perfect_match,
            "Tipo di documento": perfect_match,
            "Protocollo": perfect_match,
            "Data Protocollo": date_compare,
        },
    }

    f_metrics = flatten_dict(metrics)

    metric_values: Dict[str, bool] = {
        key: f_metrics[key](answer[key], pred_to_compare[key])
        for key in f_metrics
    }

    return metric_values


def reduce_magoh_data_eval(
    metric_values: Dict[str, bool] | Dict[str, float],
    trace=None,
) -> Union[float, bool]:
    vals = metric_values.values()
    if trace is None:
        # for now, compute the fraction of the number of valid fields over all
        # the fields
        return sum(vals) / len(vals)
    return all(vals)


def _worst_metric_value(trace=None) -> bool | float:
    if trace is None:
        return MIN_VALUE
    return False


def _worst_metric_values(trace=None):
    worst_val = _worst_metric_value(trace)
    metrics: dict[str, dict[str, float]] | dict[str, dict[str, bool]] = {
        "university": {
            "Sigla": worst_val,
            "Comune": worst_val,
            "Ubicazione": worst_val,
            "Indirizzo": worst_val,
            "Località": worst_val,
            "Data intervento": worst_val,
            "Tipo di intervento": worst_val,
            "Durata": worst_val,
            "Eseguito da": worst_val,
            "Direzione scientifica": worst_val,
            "Estensione": worst_val,
            "Numero di saggi": worst_val,
            "Profondità massima": worst_val,
            "Geologico": worst_val,
            "OGD": worst_val,
            "OGM": worst_val,
            "Profondità falda": worst_val,
        },
        "building": {
            "Istituzione": worst_val,
            "Funzionario competente": worst_val,
            "Tipo di documento": worst_val,
            "Protocollo": worst_val,
            "Data Protocollo": worst_val,
        },
    }
    f_metrics = flatten_dict(metrics)
    return f_metrics


def _is_prediction_valid(pred: Prediction) -> bool:
    """
    Reutrn None if the prediction is valid, else a metric with the worst
    value.
    """
    required_keys = set(outputStructuredDataSchema.columns.keys())
    required_keys.remove("id")
    is_valid = required_keys.issubset(set(pred.keys()))
    return is_valid


def validate_magoh_data(example: Example, pred: Prediction, trace=None):
    """
    If the prediction is not valid, then return the dict with all metric values
    at MIN_VALUE or False
    """
    if not _is_prediction_valid(pred):
        return _worst_metric_values(trace)
    return _validate_magoh_data(
        example.toDict(),
        pred.toDict(),
        trace,
    )
