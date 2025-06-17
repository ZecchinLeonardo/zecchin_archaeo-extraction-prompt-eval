from archaeo_super_prompt.dataset.load import MagohDataset
from archaeo_super_prompt.pdf_to_text.add_ocr import add_ocr_layer
from archaeo_super_prompt.pdf_to_text.extract_text import extract_text_from_pdf

print("Loading the dataset...")
myDataset = MagohDataset(6, 500)
print("Got the dataset!")
files = add_ocr_layer(myDataset.files["filepath"].to_list())
for file in files:
    print(file)
    content = extract_text_from_pdf(files[3])
print(content)
