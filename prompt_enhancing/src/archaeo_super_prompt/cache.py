from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    cast,
)
from joblib import Memory
from joblib.memory import MemorizedFunc

_CACHE_DIR = (Path(__file__).parent / "../../data/").resolve()

CacheSubpart = Literal["external", "interim", "processed"]

_memories: dict[CacheSubpart, Memory] = {
    k: Memory(str(_CACHE_DIR / k), verbose=0)
    for k in cast(Tuple[CacheSubpart, ...], ("external", "interim", "processed"))
}


def get_cache_dir_for(cache_subpart: CacheSubpart, subpart: str):
    subdir = _CACHE_DIR.joinpath(cache_subpart, subpart)
    if not subdir.exists():
        subdir.mkdir(parents=True)
    return subdir


def get_memory_for(cache_subpart: CacheSubpart):
    return _memories[cache_subpart]


## Manual caching


def identity_function[U](input: Any, output_to_be_cached: Optional[U]):
    input = input
    return output_to_be_cached


def is_input_in_the_cache(identity_function: MemorizedFunc, input: Any):
    if not identity_function.check_call_in_cache(input):
        return False
    return identity_function(input) is not None


def manually_cache_result(identity_function: MemorizedFunc, input: Any, output: Any):
    """
    Arguments:
    * identity_function: a dummy cached function to carry out the joblib cache mechanism, built from a wrapping of the identity_function function given by the module. The funtion must ignore the output argument in the caching
    """
    identity_function.call(input, output)


HashedT = TypeVar("HashedT")
Output = TypeVar("Output")
CacheIngestorFunction = Callable[[HashedT, Optional[Output]], Optional[Output]]
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
    """
    Arguments:
    * named_id_func: a function defined like this (input, output) -> output
        output can be None
    """
    identity_function = cast(
        MemorizedFunc, memory.cache(named_id_func, ignore=["output"])
    )
    cached_fn = cast(CacheIngestorFunction[HashedT, Output], identity_function)
    results: List[Tuple[Input, Optional[Output]]] = []
    inputs_to_be_processed: List[Input] = []

    for inpt in input_iter:
        hashed_inpt = input_hash_function(inpt)
        if is_input_in_the_cache(identity_function, hashed_inpt):
            results.append((inpt, cached_fn(hashed_inpt, None)))
        else:
            print(hashed_inpt)
            results.append((inpt, None))
            inputs_to_be_processed.append(inpt)

    new_results = expensive_function(iter(inputs_to_be_processed))
    for inpt, result in results:
        hashed_inpt = input_hash_function(inpt)
        if result is None:
            new_result = next(new_results)
            # just pass to this identity function to save it in the cache
            manually_cache_result(identity_function, hashed_inpt, new_result)
            yield inpt, new_result
            continue
        yield inpt, result
