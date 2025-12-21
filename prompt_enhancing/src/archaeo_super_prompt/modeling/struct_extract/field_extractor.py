"""Generic pipeline Transformer for extracting one field from featured chunks.

This transformer is a classifier which scorable and trainable.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from logging import warning
from pathlib import Path
from typing import Literal, cast, override
from pydantic import BaseModel
from pandera.typing.pandas import DataFrame
import pandas as pd
import dspy
import tqdm

import math
from typing import List, Dict, Any

# try to import tiktoken for accurate token counting, else fallback
try:
    import tiktoken
except Exception:
    tiktoken = None

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.types.intervention_id import InterventionId
from archaeo_super_prompt.types.per_intervention_feature import (
    BasePerInterventionFeatureSchema,
)
from ...types.results import ResultSchema

from . import types as extract_input_type
from ..types.detailed_evaluator import DetailedEvaluatorMixin

from . import language_model as lm_provider_mod
from .lm_vllm_client import call_vllm_generate, compute_field_confidences


# max tokens for model context (example for a 131072 token model)
DEFAULT_MODEL_CONTEXT_TOKENS = 131072
# reserve tokens for the completion result
DEFAULT_COMPLETION_TOKENS = 512

def _count_message_tokens(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Estimate tokens used by a list of chat messages.
    Uses tiktoken when available for model-specific encoding; otherwise uses
    a conservative char-based heuristic.
    """
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            # very small message framing overhead + encoded tokens
            total += 4
            for v in m.values():
                if isinstance(v, str):
                    total += len(enc.encode(v))
        return total
    # fallback heuristic: 1 token ~= 4 chars
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return math.ceil(total_chars / 4)

def _trim_messages_to_fit(messages: List[Dict[str, Any]], model: str,
                           max_completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
                           max_context_tokens: int = DEFAULT_MODEL_CONTEXT_TOKENS) -> List[Dict[str, Any]]:
    """
    Trim or summarize the longest user/system messages to fit the model context.
    This implementation truncates long 'content' strings from the earliest user messages
    (you can change strategy: e.g. summarize instead of pure truncation).
    """
    available = max_context_tokens - max_completion_tokens
    tokens = _count_message_tokens(messages, model)
    if tokens <= available:
        return messages

    # Work on a copy
    msgs = [dict(m) for m in messages]
    # sort candidate messages to trim (prefer trimming large user messages)
    candidates = [(i, len(msgs[i].get("content", ""))) for i in range(len(msgs))]
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Trim repeatedly until fit
    idx = 0
    while _count_message_tokens(msgs, model) > available and idx < len(candidates):
        i, length = candidates[idx]
        content = msgs[i].get("content", "")
        if not content:
            idx += 1
            continue
        # reduce content by half (safer than full drop)
        new_len = max(256, length // 2)
        msgs[i]["content"] = content[:new_len] + "\n\n[TRUNCATED]"
        idx += 1

    # If still too large, remove oldest non-system messages
    while _count_message_tokens(msgs, model) > available:
        # remove the earliest message that is not system (keeps system prompt if present)
        for j, m in enumerate(msgs):
            if m.get("role") != "system":
                msgs.pop(j)
                break
        else:
            break

    return msgs

# def safe_lm_call(lm_callable, *, messages: List[Dict[str, Any]], model: str, max_completion_tokens: int = DEFAULT_COMPLETION_TOKENS, **kwargs):
#     """
#     Call the LM in a safe way so the total tokens don't exceed the model's context window.
#     - trims messages to fit or
#     - if the input contains very long single documents, you may want to chunk them externally
#       and call the LM per chunk and aggregate results (not implemented here).
#     """
#     # First, try a conservative approach: reduce requested completion size if too large
#     if max_completion_tokens > 1024:
#         max_completion_tokens = 1024

#     safe_msgs = _trim_messages_to_fit(messages, model, max_completion_tokens=max_completion_tokens)
#     # attach max_tokens for completion request
#     kwargs = dict(kwargs)
#     kwargs.setdefault("max_tokens", max_completion_tokens)
#     return lm_callable(messages=safe_msgs, model=model, **kwargs)

def _safe_call_vllm_generate(
    prompt_text: str,
    model_id: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    logprobs: bool = True,
):
    """Call vllm with trimming guard. model_id is optional for backward compatibility.

    This function will trim long inputs to fit the model context window and then
    call the underlying `call_vllm_generate`. It intentionally accepts a missing
    `model_id` so existing callers that don't pass it don't raise a TypeError.
    """
    # Build chat-like messages for trimming logic
    messages = [{"role": "user", "content": prompt_text}]
    # allow model_id to be None - pass empty string to trimming heuristics in that case
    safe_messages = _trim_messages_to_fit(messages, model=(model_id or ""), max_completion_tokens=max_tokens)
    safe_prompt = " ".join(m.get("content", "") for m in safe_messages)
    # call the underlying wrapper; keep signature compatible with call_vllm_generate
    return call_vllm_generate(safe_prompt, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs)

def _wrap_model_with_confidence(model):
    """Return a callable wrapper around a dspy model that attaches
    `_confidence` and `_field_confidences` to the prediction (best-effort).

    The wrapper calls the underlying model, inspects the returned
    prediction (expected to be dict-like or dspy.Prediction), and tries to
    compute token-logprob-based confidences by calling vllm on a simple
    prompt built from the input kwargs. Failures are swallowed so behavior
    remains robust when vllm is unavailable.
    """

    class Wrapper:
        def __init__(self, inner):
            # store inner without triggering __setattr__ delegation
            object.__setattr__(self, "_inner", inner)

        def __call__(self, *args, **kwargs):
            pred = self._inner(*args, **kwargs)
            # compute a best-effort prompt from kwargs and args
            try:
                prompt_text = ""
                if kwargs:
                    prompt_text = " ".join([str(v) for v in kwargs.values() if v is not None])
                elif args:
                    prompt_text = " ".join([str(a) for a in args if a is not None])

                # normalize prediction to dict
                pred_dict = pred.toDict() if hasattr(pred, "toDict") else dict(pred)

                parsed_fields = {
                    k: "" if v is None else str(v)
                    for k, v in pred_dict.items()
                    if not str(k).startswith("_")
                }

                # call safe wrapper (backwards-compatible: model_id optional)
                resp = None
                try:
                    resp = _safe_call_vllm_generate(prompt_text, max_tokens=256, temperature=0.0, logprobs=True)
                except Exception as e:
                    # keep behavior non-fatal for the extraction pipeline but log a warning
                    warning(f"_safe_call_vllm_generate failed: {e}")

                # Validate the response contains token-level logprobs before attempting
                # to compute per-field confidences. If missing, fallback to NaN.
                if not resp:
                    warning("vllm returned empty response while computing confidences; skipping confidence computation")
                else:
                    token_logprobs = resp.get("token_logprobs")
                    tokens = resp.get("tokens")
                    if not token_logprobs or not tokens:
                        warning("vllm returned no token_logprobs/tokens; skipping compute_field_confidences")
                        try:
                            pred["_confidence"] = float("nan")
                        except Exception:
                            pass
                    else:
                        overall_conf, per_field_conf = compute_field_confidences(
                            full_text=resp.get("text", ""),
                            tokens=tokens,
                            token_logprobs=token_logprobs,
                            parsed_fields=parsed_fields,
                            fallback_to_global=True,
                        )
                        try:
                            pred["_confidence"] = float(overall_conf)
                            pred["_field_confidences"] = {k: float(v) for k, v in per_field_conf.items()}
                        except Exception:
                            pass
            except Exception:
                try:
                    # prefer numeric NaN so pandas keeps float dtype for confidence
                    pred["_confidence"] = float("nan")
                except Exception:
                    pass
            return pred

        def __getattr__(self, name):
            # Delegate attribute access to the inner model
            return getattr(self._inner, name)

        def __setattr__(self, name, value):
            # Keep _inner on the wrapper; delegate all other attributes
            if name == "_inner":
                object.__setattr__(self, name, value)
            else:
                setattr(self._inner, name, value)

    return Wrapper(model)


EvalDetailedResult = list[tuple[dspy.Example, dspy.Prediction, float]]
LLMProvider = Literal["vllm", "ollama", "openai"]


def to_prediction(output: BaseModel) -> dspy.Prediction:
    """Call this function with the pydantic-typed output for return in forward."""
    return dspy.Prediction(**output.model_dump())


def prediction_to_output[DSPyOutput](
    output_constructor: type[DSPyOutput], pred: dspy.Prediction
) -> DSPyOutput:
    """Inverse of the method above.

    Expect the prediction to be built from the _to_prediction method above
    """
    return output_constructor(**pred.toDict())


class FieldExtractor[
    DSPyInput: BaseModel,
    DSPyOutput: BaseModel,
    InputDataFrameWithKnowledge: extract_input_type.BaseInputForExtraction,
    InputDataFrameWithKnowledgeRowSchema: extract_input_type.BaseInputForExtractionRowSchema,
    DFOutput: BasePerInterventionFeatureSchema,
](
    DetailedEvaluatorMixin[
        DataFrame[InputDataFrameWithKnowledge],
        MagohDataset,
        DataFrame[ResultSchema],
    ],
    ABC,
):
    """Abstract class for extracting one field from featured chunks.

    Genericity:
    As Python does not support a lot of type checking features, the genericity
    constraints are explicited here:
    - DInput is a subtype of TypedDict, whose keys bring semantics used by \
the DSPy model as input in its forward method.
    - DOutput is a subtype of TypedDict
    - DFOutputType is a subtype of pandera.pandas.DataFrameModel
    """

    def __init__(
        self,
        llm_model_provider: LLMProvider,
        llm_model_id: str,
        llm_temperature: float,
        model: dspy.Module,
        example: tuple[DSPyInput, DSPyOutput],
        output_constructor: type[DSPyOutput],
    ) -> None:
        """Initialize the abstract class with the custom dspy module.

        Arguments:
            llm_model_provider: the service from which the llm must be fetched
            llm_model_id: the dspy chat lm to be used for the extraction 
            llm_temperature: the temperature of the llm during the prompts of \
this model
            model: the dspy module which will be used for the training and the \
inference
            example: a dspy input-output pair enabling to type check at \
runtime the genericity and also to be able to log the model in mlflow
            output_constructor: the type of the output model for building it \
generically from dictionnary expansion

        Environment variables:
            According to the llm provider, either the following env vars is
            required:
               OPENAI_API_KEY
               OLLAMA_SERVER_BASE_URL (default to http://localhost:11434)
               VLLM_SERVER_BASE_URL (default to http://localhost:8006/v1)
        """
        super().__init__()
        self.llm_model_provider: LLMProvider = llm_model_provider
        self.llm_model_id = llm_model_id
        self.llm_temperature = llm_temperature
        self._base_dspy_module = model
        self._example = example
        self._output_constructor = output_constructor

    def _infer_language_model(self):
        match self.llm_model_provider:
            case "ollama":
                return lm_provider_mod.get_ollama_model(
                    self.llm_model_id, self.llm_temperature
                )
            case "vllm":
                return lm_provider_mod.get_vllm_model(
                    self.llm_model_id, self.llm_temperature
                )
            case "openai":
                return lm_provider_mod.get_openai_model(
                    self.llm_model_id, self.llm_temperature
                )

    @classmethod
    def _itertuples(cls, X: DataFrame[InputDataFrameWithKnowledge]):
        return cast(
            Iterator[InputDataFrameWithKnowledgeRowSchema], X.itertuples()
        )

    @abstractmethod
    def _to_dspy_input(
        self,
        x: InputDataFrameWithKnowledgeRowSchema,
    ) -> DSPyInput:
        """Convert the uniformized extraction input for one intervention into one dict input for the dspy model."""
        raise NotImplementedError

    def _identity_output_set_transform_to_df(
        self, y: Iterator[tuple[InterventionId, DSPyOutput]]
    ) -> pd.DataFrame:
        """Method to directly transform the set of dspy output into a dataframe.

        Use it if needed in the transform_dspy_output implementation. For a
        type-safe usage, in your implementation, pass the output of this
        method in a scheme validation function.
        """
        return pd.DataFrame(
            [
                {
                    "id": intervention_id,
                    **dspy_output.model_dump(),
                }
                for intervention_id, dspy_output in y
            ]
        ).set_index("id")

    @abstractmethod
    def _transform_dspy_output(
        self, y: Iterator[tuple[InterventionId, DSPyOutput]]
    ) -> DataFrame[DFOutput]:
        """Transform the map of outputs into an output DataFrame with the wanted schema.

        If you want to directly use the attributes of the dspy dict output into
        the dataframe, use the _identity_output_set_transform_to_df method and
        validate this output from your DataFrameModel.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _compare_values(
        cls, predicted: DSPyOutput, expected: DSPyOutput
    ) -> tuple[float, float]:
        """Compute a metric to compare the expected output with the predicted one.

        Return:
            a score between 0 and 1
            a treshold score above which the comparison is considered as successful
        """
        raise NotImplementedError

    @override
    def fit(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
        y: MagohDataset,
        *,
        compiled_dspy_model_path: Path | None = None,
        skip_optimization=False,
        **kwargs,
    ):
        """Optimize the dspy model according to the given dataset.

        Arguments:
           X: the input dataframe with the required fields for the FieldExtractor
           y: the Magoh training dataset
           compiled_dspy_model_path: if given, a path to an already optimized dspy model, so this prompt model is directly used without reoptimize the program
           skip_optimization: if set to True, then the model is fitted with the not optimized dspy program
           kwargs: nothing usefull (just to fit the initial overriding)
        """
        kwargs = kwargs  # unused
        if skip_optimization:
            self.prompt_model_ = _wrap_model_with_confidence(self._base_dspy_module)
            return self
        if compiled_dspy_model_path is not None:
            self._base_dspy_module.load(compiled_dspy_model_path)
            self.prompt_model_ = _wrap_model_with_confidence(self._base_dspy_module)
            return self
        with dspy.settings.context(lm=self._infer_language_model()):
            # tp = dspy.MIPROv2(
            #     metric=self._dspy_metric, auto="medium", num_threads=24
            # )
            # self.prompt_model_ = _wrap_model_with_confidence(
            #     tp.compile(
            #     self._base_dspy_module,
            #     trainset=list(self._compute_devset(X, y)[1]),
            #     max_bootstrapped_demos=2,
            #     max_labeled_demos=2,
            #     requires_permission_to_run=False,
            # )
            # )
            tp = dspy.MIPROv2(metric=self._dspy_metric, auto="medium", num_threads=24)
            # compute trainset once and guard against too-small trainset (dspy requires >=2 examples)
            trainset = list(self._compute_devset(X, y)[1])
            if len(trainset) < 2:
                # Not enough examples to optimize: fallback to base module without optimization.
                warning(
                    "Not enough examples to optimize DSPy program (need >=2). "
                    "Falling back to base dspy module (no compile)."
                )
                self.prompt_model_ = _wrap_model_with_confidence(self._base_dspy_module)
                return self
            self.prompt_model_ = _wrap_model_with_confidence(
                tp.compile(
                    self._base_dspy_module,
                    trainset=trainset,
                )
            )
        return self

    def _typed_forward(self, inpt: DSPyInput) -> DSPyOutput:
        """Carry out a type safe forward on the dspy module."""
        # Call the dspy model and keep the raw prediction so we can attach
        # best-effort confidence metadata computed from vllm token logprobs.
        raw_pred = cast(dspy.Prediction, self.prompt_model_(**inpt.model_dump()))

        # Best-effort: compute confidence from vllm token logprobs if the
        # prediction does not already contain `_confidence` (the prompt model
        # may itself be wrapped to provide this metadata).
        try:
            # build a prompt-ish text from the input model fields
            inp_map = inpt.model_dump()
            prompt_text = " ".join([str(v) for v in inp_map.values() if v is not None])
            pred_dict = raw_pred.toDict() if hasattr(raw_pred, "toDict") else dict(raw_pred)
            if "_confidence" not in pred_dict or pred_dict.get("_confidence") is None:
                # call safe wrapper passing model id when available
                resp = None
                try:
                    resp = _safe_call_vllm_generate(prompt_text, model_id=self.llm_model_id, max_tokens=256, temperature=0.0, logprobs=True)
                except Exception as e:
                    warning(f"_safe_call_vllm_generate failed in _typed_forward: {e}")

                parsed_fields = {
                    k: "" if v is None else str(v)
                    for k, v in pred_dict.items()
                    if not str(k).startswith("_")
                }

                if not resp:
                    # no response: fallback to NaN
                    try:
                        raw_pred["_confidence"] = float("nan")
                    except Exception:
                        pass
                else:
                    tokens = resp.get("tokens")
                    token_logprobs = resp.get("token_logprobs")
                    if not tokens or not token_logprobs:
                        warning("vllm returned no token_logprobs/tokens in _typed_forward; skipping compute_field_confidences")
                        try:
                            raw_pred["_confidence"] = float("nan")
                        except Exception:
                            pass
                    else:
                        overall_conf, per_field_conf = compute_field_confidences(
                            full_text=resp.get("text", ""),
                            tokens=tokens,
                            token_logprobs=token_logprobs,
                            parsed_fields=parsed_fields,
                            fallback_to_global=True,
                        )
                        try:
                            raw_pred["_confidence"] = float(overall_conf)
                            raw_pred["_field_confidences"] = {k: float(v) for k, v in per_field_conf.items()}
                        except Exception:
                            pass
        except Exception:
            # If vllm not reachable or any error occurs, do not break the forward
            try:
                # use numeric NaN so pandas will treat column as float
                raw_pred["_confidence"] = float("nan")
            except Exception:
                pass

        return prediction_to_output(self._output_constructor, raw_pred)

    @override
    def predict(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
    ) -> DataFrame[DFOutput]:
        """Generic transform operation."""
        inputs = (
            (InterventionId(row.Index), self._to_dspy_input(row))
            for row in self._itertuples(X)
        )
        with dspy.settings.context(lm=self._infer_language_model()):
            return self._transform_dspy_output(
                (
                    intervention_id,
                    self._typed_forward(inpt),
                )
                for intervention_id, inpt in tqdm.tqdm(
                    inputs,
                    total=len(X),
                    desc="Field extraction",
                    unit="processed intervention",
                )
            )

    @classmethod
    @abstractmethod
    def filter_training_dataset(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> set[InterventionId]:
        """Among the given set of intervention records, select only those with suitable answers for a training or an evaluation."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _select_answers(
        cls, y: MagohDataset, ids: set[InterventionId]
    ) -> dict[InterventionId, DSPyOutput]:
        raise NotImplementedError

    def _compute_devset(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
        y: MagohDataset,
    ) -> tuple[tuple[int, ...], tuple[dspy.Example, ...]]:
        good_ids = self.filter_training_dataset(
            y, set(InterventionId(id_) for id_ in list(X.index))
        )
        not_good_ids = X[~(X.index.isin(good_ids))].index.to_list()
        if not_good_ids:
            warning(
                f"These records will not be used in the devset, as their answers are incorrect: {not_good_ids}"
            )
        inputs = {
            InterventionId(row.Index): self._to_dspy_input(row)
            for row in self._itertuples(X[X.index.isin(good_ids)])
        }
        answers = self._select_answers(y, set(inputs.keys()))

        # Only use ids present in BOTH inputs and answers (avoid KeyError)
        valid_ids = [id_ for id_ in inputs if id_ in answers]
        if not valid_ids:
            return tuple(), tuple()

        kept_ids, examples = zip(
            *(
                (
                    id_,
                    (
                        lambda model_input: dspy.Example(
                            **model_input,
                            # TODO: select only one field with an abstract
                            **answers[id_].model_dump(),
                        ).with_inputs(*model_input.keys())
                    )(inputs[id_].model_dump()),
                )
                # for id_ in inputs.keys()
                for id_ in valid_ids
            )
        )
        return kept_ids, examples

    def _dspy_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> float | bool:
        result, passable_treshold = self._compare_values(
            self._output_constructor(**prediction.toDict()),
            self._output_constructor(**example.toDict()),
        )
        if trace is None:
            return result
        return result >= passable_treshold

    @override
    def score(
        self,
        X: DataFrame[InputDataFrameWithKnowledge],
        y: MagohDataset,
        sample_weight=None,
    ):
        """Run a local evaluation of the dpsy model over the given X dataset.

        Also save the per-field results for each test record in a cached
        dataframe, accessible after the function call with the score_results
        property (it will not equal None after a sucessful run of this method)

        To fit the sklearn Classifier interface, this method return a reduced
        floating metric value for the model.
        """
        sample_weight = sample_weight  # unused

        _, devset = self._compute_devset(X, y)

        with dspy.settings.context(lm=self._infer_language_model()):
            evaluator = dspy.Evaluate(
                devset=list(devset[1]),
                metric=self._dspy_metric,
                return_outputs=False,
                provide_traceback=True,  # TODO: remove it for traceback
                num_threads=1,  # TODO: set it
                display_progress=True,
                display_table=5,
            )
            score = cast(float, evaluator(self.prompt_model_))
        return score

    @override
    def score_and_transform(self, X, y):
        kept_ids, devset = self._compute_devset(X, y)
        with dspy.settings.context(lm=self._infer_language_model()):
            evaluator = dspy.Evaluate(
                devset=list(devset),
                metric=self._dspy_metric,
                return_outputs=True,
                provide_traceback=True,  # TODO: remove it for traceback
                num_threads=1,  # TODO: set it
                display_progress=True,
                display_table=5,
            )
            score, score_table = cast(
                tuple[float, EvalDetailedResult],
                evaluator(self.prompt_model_),
            )
            # return score, ResultSchema.validate(
            #     pd.DataFrame(
            #         [
            #             {
            #                 "id": id_,
            #                 # "field_name": self.field_to_be_extracted(),
            #                 "field_name": (
            #                     ",".join(self.field_to_be_extracted())
            #                     if isinstance(self.field_to_be_extracted(), tuple)
            #                     else self.field_to_be_extracted()
            #                 ),
            #                 "metric_value": score,
            #                 # TODO: specify the evaluation method
            #                 "evaluation_method": "not specified yet",
            #                 # Filter out metadata keys (private keys starting with '_') from
            #                 # predicted values so we don't try to lookup them in the example
            #                 # dict (which causes KeyError).
            #                 "expected_value": {
            #                     k: ex_dict.get(k) for k in pred_dict.keys() if not str(k).startswith("_")
            #                 },
            #                 "predicted_value": {
            #                     k: pred_dict[k] for k in pred_dict.keys() if not str(k).startswith("_")
            #                 },
            #                 # Best-effort: expose overall confidence and per-field confidences
            #                 # if available in the prediction under the reserved keys.
            #                 "confidence": pred_dict.get("_confidence") if "_confidence" in pred_dict else None,
            #                 # "field_confidences": pred_dict.get("_field_confidences") if "_field_confidences" in pred_dict else None,
            #             }
            #             for id_, (ex_dict, pred_dict, score) in zip(
            #                 kept_ids,
            #                 (
            #                     (ex.toDict(), pred.toDict(), score)
            #                     for ex, pred, score in score_table
            #                 ),
            #             )
            #         ]
            #     ),
            #     lazy=True,
            # )
            rows = [
                {
                    "id": id_,
                    "field_name": (
                        ",".join(self.field_to_be_extracted())
                        if isinstance(self.field_to_be_extracted(), tuple)
                        else self.field_to_be_extracted()
                    ),
                    "metric_value": score_value,
                    "evaluation_method": "not specified yet",
                    "expected_value": {
                        k: ex_dict.get(k) for k in pred_dict.keys() if not str(k).startswith("_")
                    },
                    "predicted_value": {
                        k: pred_dict[k] for k in pred_dict.keys() if not str(k).startswith("_")
                    },
                    "confidence": pred_dict.get("_confidence") if "_confidence" in pred_dict else None,
                }
                for id_, (ex_dict, pred_dict, score_value) in zip(
                    kept_ids,
                    ((ex.toDict(), pred.toDict(), score_) for ex, pred, score_ in score_table),
                )
            ]
            df_results = pd.DataFrame(rows)
            # ensure numeric columns have the expected dtype for pandera validation
            if "metric_value" in df_results.columns:
                df_results["metric_value"] = df_results["metric_value"].astype("float64")
            if "confidence" in df_results.columns:
                # coerce None/objects to NaN and cast to float64
                df_results["confidence"] = pd.to_numeric(df_results["confidence"], errors="coerce").astype("float64")
            return score, ResultSchema.validate(df_results, lazy=True)

    @staticmethod
    @abstractmethod
    def field_to_be_extracted() -> str:
        """A human label/description of the field related to the Extractor."""
        raise NotImplementedError

    @property
    def signature_example(self):
        """Return an example of input/output dict pair for the dspy model.

        This property is usefull for a logging by mlflow.
        """
        return self._example

    @property
    def lm(self):
        """Return the llm model."""
        return self._infer_language_model()
