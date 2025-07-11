from pathlib import Path
from typing import Any, Literal, Optional, Tuple, cast
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
