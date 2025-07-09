from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from . import inference

app = FastAPI()


class Batch(BaseModel):
    chunks: List[str]


@app.post("/ner/")
async def ner_inference(batch: Batch) -> List[List[inference.NerOutput]]:
    return inference.infer(batch.chunks)


@app.get("/")
async def root():
    return {"message": "Inference server ready!"}
