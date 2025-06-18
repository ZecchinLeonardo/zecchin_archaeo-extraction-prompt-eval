from typing import List, TypedDict, cast
import numpy as np
import numpy.typing as npt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str):
    # Minuscole, rimozione punteggiatura
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).strip()


class SoftAccuracyResult(TypedDict):
    accuracy: np.floating
    similarities: npt.NDArray[np.floating]
    matches: npt.NDArray[np.bool]


def soft_accuracy(
    predictions: List[str], references: List[str], threshold=0.75
) -> SoftAccuracyResult:
    assert len(predictions) == len(references), "Liste di lunghezza diversa!"

    preproc_preds = [preprocess(p) for p in predictions]
    preproc_refs = [preprocess(r) for r in references]

    all_texts = preproc_preds + preproc_refs
    vectorizer = TfidfVectorizer().fit(all_texts)

    pred_vecs = vectorizer.transform(preproc_preds)
    ref_vecs = vectorizer.transform(preproc_refs)

    similarities = cast(np.ndarray, cosine_similarity(pred_vecs, ref_vecs)).diagonal()
    correct = similarities >= threshold
    return {
        "accuracy": np.mean(correct),
        "similarities": similarities,
        "matches": correct,
    }


def example():
    preds = [
        "mura di Lucca",
        "chiesa di Sant Ambrogio, in corrispondenza delle mura di Lucca",
        "Roma",
    ]
    refs = ["Lucca", "mura di Lucca", "Roma"]

    result = soft_accuracy(preds, refs, threshold=0.7)
    print(f"Accuratezza soft: {result['accuracy']:.2f}")
    print("Similarità:", result["similarities"])
