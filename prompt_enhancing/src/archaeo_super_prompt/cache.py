from pathlib import Path
from typing import Literal
from joblib import Memory

_CACHE_DIR = Path(__file__) / "../../data/"

CacheSubpart = Literal["external", "interim", "processed"]

_memories: dict[CacheSubpart, Memory] = {
    k: Memory(str(_CACHE_DIR / k), verbose=0) for k in ("external", "interim", "processed")
}


def get_cache_dir_for(cache_subpart: CacheSubpart, subpart: str):
    subdir = _CACHE_DIR.joinpath(cache_subpart, subpart)
    if not subdir.exists():
        subdir.mkdir(parents=True)
    return subdir


def get_memory_for(cache_subpart: CacheSubpart):
    return _memories[cache_subpart]
