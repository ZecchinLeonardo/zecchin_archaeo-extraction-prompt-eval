from functools import reduce
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, cast
from dspy import Example, Prediction
import dspy
from dspy.evaluate.metrics import answer_exact_match
import mlflow

from .display_fields import add_to_arrays
from .smart_match_checking import check_with_LLM

from ..magoh_target import MagohData, toMagohData

from ..models.main_pipeline import ExtractedInterventionData

MIN_VALUE = 0
MAX_VALUE = 1

U, V = TypeVar("U"), TypeVar("V")


def validate_magoh_data(answer: MagohData, pred: ExtractedInterventionData, trace=None):
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
        return check_with_LLM(e, p, trace) == 1

    @validate_type
    def neutral(e: str, p: str):
        e = e
        p = p
        return True

    @validate_type
    def date_compare(e: str, p: str):
        # TODO:
        return answer_exact_match(
            dspy.Example(answer=e), dspy.Prediction(answer=p), trace
        )

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
            "Estensione": neutral,
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
                answer[first_depth_key][second_detph_key],
                pred_to_compare[first_depth_key][second_detph_key],
            )
            for second_detph_key in metrics[first_depth_key]
        }
        for first_depth_key in metrics
    }
    return metric_values


def _validated_json(
    answer: MagohData,
    pred: ExtractedInterventionData,
    run: mlflow.ActiveRun,
    trace=None,
) -> Union[float, bool]:
    metric_values = validate_magoh_data(answer, pred, trace)
    add_to_arrays(answer, pred, metric_values, run)

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


def validated_json(run: mlflow.ActiveRun):
    def validated_json(
        example: Example, pred: Prediction, trace=None
    ) -> Union[float, bool]:
        pred_dict = pred.toDict()
        if not pred_dict:
            return MIN_VALUE if trace is None else False
        return _validated_json(
            cast(MagohData, example.answer),
            cast(ExtractedInterventionData, pred_dict),
            run,
            trace,
        )
    return validated_json
