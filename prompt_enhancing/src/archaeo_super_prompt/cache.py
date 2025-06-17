from pathlib import Path
from joblib import Memory

memory = Memory(str(Path("./.cache")), verbose=0)
