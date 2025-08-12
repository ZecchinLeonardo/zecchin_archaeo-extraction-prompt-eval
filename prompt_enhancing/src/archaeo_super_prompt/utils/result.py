"""Utils for saving or loading plots from results."""

from pathlib import Path

__report_dir = (Path(__file__).parent / "../../../reports/").resolve()
__compiled_models_dir = (Path(__file__).parent / "../../../models/compiled/").resolve()

def get_report_dir() -> Path:
    """Return the project's directory with the plots and the results of the experiments."""
    return __report_dir

def get_model_store_dir() -> Path:
    """Return the project's directory with the compiled dspy models in json."""
    return __compiled_models_dir
