import numpy as np
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
from typing import cast, List, Callable, Dict, Any

tokenizer = AutoTokenizer.from_pretrained(
    "DeepMount00/Italian_NER_XXL", torch_dtype="auto"
)
model = AutoModelForTokenClassification.from_pretrained(
    "DeepMount00/Italian_NER_XXL",
    ignore_mismatched_sizes=True,
    torch_dtype="auto",
)

# set CUDA_VISIBLE_DEVICES, so 0 index will be mapped to an allowed GPU
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=-1)
example = """Il commendatore Gianluigi Alberico De Laurentis-Ponti, con residenza legale in Corso Imperatrice 67,  Torino, avente codice fiscale DLNGGL60B01L219P, è amministratore delegato della "De Laurentis Advanced Engineering Group S.p.A.",  che si trova in Piazza Affari 32, Milano (MI); con una partita IVA di 09876543210, la società è stata recentemente incaricata  di sviluppare una nuova linea di componenti aerospaziali per il progetto internazionale di esplorazione di Marte."""


def update_field(d: Dict, key: str, func: Callable[[Any], Any]):
    return {**d, key: func(d[key])}


def no_numpy_float(np_float: np.float32) -> float:
    return np_float.item()

class NerOutput(BaseModel):
    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


def infer(input_text: List[str]):
    entities_for_batch = nlp(input_text)
    return [
        [
            cast(NerOutput, update_field(entity, "score", no_numpy_float))
            for entity in entities_of_element
        ]
        for entities_of_element in entities_for_batch
    ]
