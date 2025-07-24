"""Cache feature management."""

from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)
from collections.abc import Callable, Iterator
from joblib import Memory
from joblib.memory import MemorizedFunc

_CACHE_DIR = (Path(__file__).parent / "../../../data/").resolve()

CacheSubpart = Literal["external", "interim", "processed", "raw"]

_memories: dict[CacheSubpart, Memory] = {
    k: Memory(str(_CACHE_DIR / k), verbose=0)
    for k in cast(
        tuple[CacheSubpart, ...], ("external", "interim", "processed")
    )
}


def get_cache_dir_for(cache_subpart: CacheSubpart, subpart: str):
    """Return a path object pointing to a subdir of the given "/data" directory."""
    subdir = _CACHE_DIR.joinpath(cache_subpart, subpart)
    if not subdir.exists():
        subdir.mkdir(parents=True)
    return subdir


def get_memory_for(cache_subpart: CacheSubpart):
    """Get the joblib cache memory related to a subpath of the "/data" directory."""
    return _memories[cache_subpart]


## Manual caching


def identity_function(input: Any, output_to_be_cached: Any | None):
    """Identity function."""
    input = input
    return output_to_be_cached


def is_input_in_the_cache(identity_function: MemorizedFunc, input: Any):
    """Return if the input has already an output saved in the cache."""
    if not identity_function.check_call_in_cache(input):
        return False
    return identity_function(input) is not None


def manually_cache_result(
    identity_function: MemorizedFunc, input: Any, output: Any
):
    """Manually save the input and its output in the joblib's cache.

    Arguments:
        identity_function: a dummy cached function to carry out the joblib \
cache mechanism, built from a wrapping of the identity_function function given \
by the module. The funtion must ignore the output argument in the caching
        input: a hashable input
        output: the value to be saved in the cache
    """
    identity_function.call(input, output)


HashedT = TypeVar("HashedT")
Output = TypeVar("Output")
CacheIngestorFunction = Callable[[HashedT, Output | None], Output | None]
"""The name of the arguments is important: the output argument must be exactly
named 'output'
"""


def escape_expensive_run_when_cached[Input, HashedT, Output](
    named_id_func: CacheIngestorFunction[HashedT, Output],
    memory: Memory,
    input_hash_function: Callable[[Input], HashedT],
    expensive_function: Callable[[Iterator[Input]], Iterator[Output]],
    input_iter: Iterator[Input],
):
    """TODO: comment.

    Arguments:
        named_id_func: a function defined like this (input, output) -> output \
output can be None
        memory: TODO
        input_hash_function: TODO
        expensive_function: TODO
        input_iter: TODO
    """
    identity_function = cast(
        MemorizedFunc, memory.cache(named_id_func, ignore=["output"])
    )
    cached_fn = cast(CacheIngestorFunction[HashedT, Output], identity_function)
    results: list[tuple[Input, Output | None]] = []
    inputs_to_be_processed: list[Input] = []

    for inpt in input_iter:
        hashed_inpt = input_hash_function(inpt)
        result = (
            cached_fn(hashed_inpt, None)
            if is_input_in_the_cache(identity_function, hashed_inpt)
            else None
        )
        results.append((inpt, result))
        if result is None:
            inputs_to_be_processed.append(inpt)

    new_results = expensive_function(iter(inputs_to_be_processed))
    for inpt, result in results:
        if result is None:
            hashed_inpt = input_hash_function(inpt)
            try:
                new_result = next(new_results)
                # just pass to this identity function to save it in the cache
                manually_cache_result(
                    identity_function, hashed_inpt, new_result
                )
                yield inpt, new_result
                continue
            except StopIteration:
                raise Exception(
                    f"The function {named_id_func.__name__} has missed some results to be produced"
                )
        yield inpt, result


def manualy_cache_batch_processing[Input, Output](
    path_for_input: Callable[[Input], Path],
    cache_on_disk: Callable[[Output, Path], None],
    load_output_from_cache: Callable[[Path], Output],
    expensive_function: Callable[[Iterator[Input]], Iterator[Output]],
    input_iter: Iterator[Input],
) -> Iterator[tuple[Input, Output]]:
    """Lazily execute an expensive function taking a batch of inputs with cache.

    Execute an expensive function taking a batch of inputs, with escaping
    all the inputs of the batch whose the output is already saved in the cache.
    """
    results_from_current_cache_only = [
        (inpt, (lambda: load_output_from_cache(p)) if p.exists() else None)
        for inpt, p in map(
            lambda inpt: (inpt, path_for_input(inpt)), input_iter
        )
    ]
    not_cached_yet_inputs = (
        inpt
        for inpt, opt_output in results_from_current_cache_only
        if opt_output is None
    )
    new_results = expensive_function(not_cached_yet_inputs)

    def put_in_cache_and_return(input: Input, output: Output):
        cache_on_disk(output, path_for_input(input))
        return output

    return (
        (
            inpt,
            opt_output()
            if opt_output is not None
            else put_in_cache_and_return(inpt, next(new_results)),
        )
        for inpt, opt_output in results_from_current_cache_only
    )
