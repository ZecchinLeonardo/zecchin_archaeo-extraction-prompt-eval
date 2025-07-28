import numpy as np
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
from typing import cast, List, Callable, Dict, Any
from os import getenv

def get_devices():
    """Get the desired gpu identifiers from a env variable."""
    device = getenv("CUDA_VISIBLE_DEVICES")
    if device is None:
        raise EnvironmentError("Missing the env var MAGOH_GPU_DEVICES")
    if "," in device:
        return device
    if not device.isdigit():
        raise EnvironmentError("You must precise an integer as device.")
    return int(device)

tokenizer = AutoTokenizer.from_pretrained(
    "DeepMount00/Italian_NER_XXL", torch_dtype="auto"
)
model = AutoModelForTokenClassification.from_pretrained(
    "DeepMount00/Italian_NER_XXL",
    ignore_mismatched_sizes=True,
    torch_dtype="auto",
)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=get_devices())
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
