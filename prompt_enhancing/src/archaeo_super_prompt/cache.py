from pathlib import Path
from joblib import Memory

_CACHE_DIR = Path(".cache/")

def get_cache_subdir(subdir_name: str):
    subdir = _CACHE_DIR.joinpath(subdir_name)
    if not subdir.exists():
        subdir.mkdir(parents=True)
    return subdir

memory = Memory(str(_CACHE_DIR), verbose=0)

