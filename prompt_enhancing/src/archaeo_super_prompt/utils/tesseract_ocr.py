# Replace scanner with a Tesseract-based scanner

import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

class TesseractPreprocessing:
    def __init__(self, lang='ita', incipit_only=True, max_lines=10):#, pages=None):
        self.lang = lang
        self.incipit_only = incipit_only
        self.max_lines = max_lines
        # self.pages = pages

    def _image_to_text(self, img):
        return pytesseract.image_to_string(img, lang=self.lang)

    def transform(self, df, pages=None):
        records = []
        for _, row in df.iterrows():
            # try common path column names
            path = None
            for key in ("path", "file", "filepath", "filename"):
                if key in row and pd.notna(row[key]):
                    path = Path(row[key])
                    break
            if path is None:
                # maybe the row itself is a path string
                candidate = row.iloc[0] if len(row) > 0 else None
                if isinstance(candidate, (str, Path)):
                    path = Path(candidate)
            if path is None:
                continue

            text = ""
            try:
                if path.suffix.lower() == ".pdf":
                #     pages = convert_from_path(str(path), first_page=1, last_page=self.pages)
                #     if pages:
                #         text = self._image_to_text(pages[0])
                # else:
                #     img = Image.open(path)
                #     text = self._image_to_text(img)
                    if pages is None and self.incipit_only == True:
                        pages_imgs = convert_from_path(str(path), first_page=1, last_page=1)
                    elif pages is None and self.incipit_only == False:
                        pages_imgs = convert_from_path(str(path))
                    else:
                        pages_imgs = convert_from_path(str(path), first_page=1, last_page=pages)
                        
                    if pages_imgs:
                        # join OCR from all pages
                        text = "\n\n".join(self._image_to_text(p) for p in pages_imgs)
                else:
                    img = Image.open(path)
                    text = self._image_to_text(img)
            except Exception as e:
                text = f"[OCR ERROR: {e}]"

            if self.incipit_only:
                lines = [line for line in text.splitlines() if line.strip()]
                text = "\n".join(lines[: self.max_lines])

            records.append({
                "id": row.get("id", None),
                "text": text,
                "source": str(path),
            })

        return pd.DataFrame.from_records(records)