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


def score_dag(
    parts: ExtractionDAGParts, inputs: PDFPathDataset, eval_ds: MagohDataset
):
    """From an already fitted model, apply scoring over the extractors."""
    preprocessed_input = parts.preprocessing_root.make_dag().transform(inputs)

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

