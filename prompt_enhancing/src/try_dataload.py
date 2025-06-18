import functools
from pathlib import Path
from typing import Dict, List, cast

from dspy import dspy
from pandas import Series

from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.evaluation.load_examples import DevSet
from archaeo_super_prompt.models.main_pipeline import ExtractDataFromInterventionReport
from archaeo_super_prompt.pdf_to_text.add_ocr import add_ocr_layer
from archaeo_super_prompt.pdf_to_text.extract_text import extract_text_from_pdf

from sklearn.pipeline import Pipeline

from archaeo_super_prompt.language_model import load_model
from archaeo_super_prompt.signatures.input import Filename, PDFSources

Id = int


class OCR_Transformer:
    def fit(self, X: Dict[Id, List[Path]], y=None):
        X = X
        y = y
        return self

    def transform(self, X: Dict[Id, List[Path]]) -> Dict[Id, List[Path]]:
        id_list = list(X.keys())
        flatten = functools.reduce(
            lambda acc_l, id_: acc_l + X[id_], id_list, cast(List[Path], [])
        )
        output_paths = add_ocr_layer(flatten)

        def rebuild_dictionary(
            acc: Dict[Id, List[Path]],
            id_list: List[Id],
            remaining_output_paths: List[Path],
        ):
            if not id_list:
                return acc
            id_ = id_list.pop(0)
            item_to_pick_nb = len(X[id_])
            acc[id_] = remaining_output_paths[:item_to_pick_nb]
            return rebuild_dictionary(
                acc, id_list, remaining_output_paths[item_to_pick_nb:]
            )

        return rebuild_dictionary({}, id_list, output_paths)


class TextExtractor:
    def fit(self, X: Dict[Id, List[Path]], targets: MagohDataset):
        X = X
        targets = targets
        return self

    def transform(self, X: Dict[Id, List[Path]]):
        return {
            id_: {cast(Filename, p.name): extract_text_from_pdf(p) for p in X[id_]}
            for id_ in X
        }


class MagohDataExtractor:
    def __init__(self, llm_temp=0.0) -> None:
        self._module = ExtractDataFromInterventionReport()
        self._llm = load_model(llm_temp)
        self._module.set_lm(self._llm)

    def fit(self, X: Dict[Id, PDFSources], Y: MagohDataset):
        # TODO: call a dspy optimizer
        X = X
        Y = Y
        return self

    def transform(self, X: Dict[Id, PDFSources]):
        return {id_: self._module.forward_and_type(X[id_]) for id_ in X}

    def score(self, X: Dict[Id, PDFSources], targets: MagohDataset):
        # TODO: pass
        devset: DevSet = [
            dspy.Example(
                document_ocr_scan=X[id_], answer=targets.get_answer(id_)
            ).with_inputs("document_ocr_scan")
            for id_ in X
        ]
        


def inputs_from_dataset(dataset: MagohDataset) -> Dict[Id, List[Path]]:
    return {
        id_: cast(
            Series,
            dataset.files[dataset.files["scheda_intervento.id"] == id_]["filepath"],
        ).to_list()
        for id_ in cast(Series, dataset.intervention_data["scheda_intervento.id"])
    }


# TODO: better manage the cache
pipeline = Pipeline(
    [
        ("ocr", OCR_Transformer()),
        ("pdf_reader", TextExtractor()),
        # ("extractor", MagohDataExtractor()),
    ]
)

print("Loading the dataset...")
myDataset = MagohDataset(6, 500)
print("Got the dataset!")
inputs = inputs_from_dataset(myDataset)
MagohDataExtractor().score(pipeline.fit_transform(inputs), myDataset)
