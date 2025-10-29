"""Code containing the global model and a way to infer into it."""

from ..dataset.load import MagohDataset
from ..types.pdfpaths import PDFPathDataset
from .train import ExtractionDAGParts
from functools import reduce
import pandas as pd


def build_complete_inference_dag(parts: ExtractionDAGParts):
    """Build the inference model from fitted parts."""
    return (
        reduce(
            lambda acc, item: acc.add_node(item[0], [item[1]]),
            parts.extraction_parts,
            parts.preprocessing_root,
        )
        .add_node(*parts.final_component)
        .make_dag()
    )


def _normalize_inputs_df(inputs):
    """
    Ensure inputs is a pandas DataFrame having an 'id' column:
    - if 'id' is the index, reset_index
    - strip whitespace from column names
    - if the first column is an 'Unnamed' index that looks like 0..n, drop it
    """
    if not isinstance(inputs, pd.DataFrame):
        try:
            inputs = pd.DataFrame(inputs)
        except Exception:
            raise ValueError("score_dag expected a pandas DataFrame or convertible object for `inputs`.")

    # strip column names
    inputs.columns = inputs.columns.astype(str).str.strip()

    # if id is index, reset it to column
    if inputs.index.name == "id":
        inputs = inputs.reset_index()

    # drop a leading saved index column if it looks numeric 0..n
    first_col = inputs.columns[0] if len(inputs.columns) > 0 else None
    if first_col and (first_col.startswith("Unnamed") or first_col == ""):
        # sample values to test if this is a saved index
        sample = inputs.iloc[:, 0].dropna().astype(str).head(20).tolist()
        if sample and all(s.isdigit() for s in sample):
            inputs = inputs.iloc[:, 1:].copy()
            inputs.columns = inputs.columns.astype(str).str.strip()

    # final check
    if "id" not in inputs.columns:
        raise ValueError(
            "score_dag requires `inputs` DataFrame to contain an 'id' column.\n"
            f"Columns present: {list(inputs.columns)}\n"
            "If your CSV was saved from pandas, ensure it was written with index=False, "
            "or re-load using a safe loader that drops a saved index column.\n"
            f"Example head:\n{inputs.head(5).to_string(index=False)}"
        )

    # coerce id to integer-like where appropriate (but keep missing as NA)
    try:
        inputs["id"] = pd.to_numeric(inputs["id"].astype(str).str.strip(), errors="coerce").astype("Int64")
    except Exception:
        # fall back to leaving as-is if coercion fails
        pass

    return inputs

def score_dag(
    parts: ExtractionDAGParts, inputs: PDFPathDataset, eval_ds: MagohDataset
):
    # normalize & validate inputs before feeding to preprocessing DAG
    inputs = _normalize_inputs_df(inputs)

     # run the preprocessing DAG but catch common errors and re-raise with diagnostics
    try:
        preprocessed_input = parts.preprocessing_root.make_dag().transform(inputs)
    except KeyError as e:
        # Provide helpful diagnostics instead of raw pandas KeyError deeper in pipeline
        raise RuntimeError(
            "Preprocessing DAG failed with KeyError: likely a missing column in the "
            "DataFrame produced by a transformer (or empty lookup result). "
            "Inputs supplied to the DAG are shown below for debugging.\n\n"
            f"Original inputs columns: {list(inputs.columns)}\n"
            f"Original inputs shape: {inputs.shape}\n"
            f"First rows:\n{inputs.head(5).to_string(index=False)}\n\n"
            f"Transformer error: {e}"
        ) from e
    
    """From an already fitted model, apply scoring over the extractors."""
    # preprocessed_input = parts.preprocessing_root.make_dag().transform(inputs)

    scores = [
        (
            extract_id,
            *fe.score_and_transform(
                preprocessed_input[dep.component_id], eval_ds
            ),
        )
        for (extract_id, fe), dep in parts.extraction_parts
        if not isinstance(fe, str)
    ]
    return {extract_id: score for extract_id, score, _ in scores}, pd.concat(
        df for _, _, df in scores
    )

