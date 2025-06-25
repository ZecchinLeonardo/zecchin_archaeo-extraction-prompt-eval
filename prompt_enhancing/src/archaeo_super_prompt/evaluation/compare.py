from functools import reduce
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, cast
from dspy import Example, Prediction
import dspy
from dspy.evaluate.metrics import answer_exact_match

from .similarity_match import soft_accuracy

from .smart_match_checking import check_date_with_LLM, check_with_LLM

from ..magoh_target import MagohData, toMagohData

from ..models.main_pipeline import ExtractedInterventionData

MIN_VALUE = 0
MAX_VALUE = 1

U, V = TypeVar("U"), TypeVar("V")


def _validate_magoh_data(
    answer: MagohData, pred: ExtractedInterventionData, trace=None
):
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
        def inner(e: Union[U, List[U], Tuple[U]], p: Union[U, List[U], Tuple[U]]):
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
        llm_check = check_with_LLM(TRESHOLD)
        similarity_check = soft_accuracy([e], [p], TRESHOLD)["matches"].item()
        return similarity_check
        return llm_check(e, p, trace) == 1

    @validate_type
    def neutral(e: str, p: str):
        e = e
        p = p
        return False

    @validate_type
    def date_compare(e: str, p: str):
        # TODO: avoid LLM when the typing will be better
        return check_date_with_LLM(e, p, trace) == 1

    pred_to_compare = toMagohData(pred)
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

    metric_values: Dict[str, Dict[str, bool]] = {
        first_depth_key: {
            second_detph_key: metrics[first_depth_key][second_detph_key](
                answer[first_depth_key][second_detph_key],  # type: ignore
                pred_to_compare[first_depth_key][second_detph_key],  # type: ignore
            )
            for second_detph_key in metrics[first_depth_key]
        }
        for first_depth_key in metrics
    }
    return metric_values


def reduce_magoh_data_eval(
    metric_values: Dict[str, Dict[str, bool]],
    trace=None,
) -> Union[float, bool]:
    if trace is None:
        # for now, compute the fraction of the number of valid fields over all
        # the fields
        summed_metrics, property_nb = reduce(
            lambda t, dico: (t[0] + sum(dico.values()), t[1] + len(dico)),
            metric_values.values(),
            (0, 0),
        )
        return summed_metrics / property_nb
    return all(
        reduce(
            lambda flat, vals: flat + list(vals.values()), metric_values.values(), []
        )
    )


def is_prediction_valid(pred: Prediction, trace=None):
    """
    Reutrn None if the prediction is valid, else a metric with the worst
    value.
    """
    pred_dict = pred.toDict()
    if not pred_dict:
        return MIN_VALUE if trace is None else False
    return None


def validate_magoh_data(example: Example, pred: Prediction, trace=None):
    """
    Assume the given prediction is valid (check it with is_prediction_valid)
    """
    return _validate_magoh_data(
        {
            "university": example.university,
            "building": example.building,
            "scheda_intervento": {"id": 0},
        },
        cast(ExtractedInterventionData, pred),
        trace,
    )
