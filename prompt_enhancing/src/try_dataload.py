from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.pdf_to_text.add_ocr import add_ocr_layer

print("Loading the dataset...")
myDataset = MagohDataset(6, 500)
print("Got the dataset!")
add_ocr_layer(myDataset.files["filepath"].to_list())
