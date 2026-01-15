import fitz
import pdfplumber
import pytesseract
from PIL import Image
import json
import os
import io
from config import PDF_PATH, OUTPUT_PATH, TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# extracting text from the pdf file
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    results = []  # add text extracted from each page
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            results.append({
                "content": text,
                "modality": "text",
                "page": page_num
            })
    return results


# extracting tables from the pdf file
def extract_tables(pdf_path):
    results = []   # add tables extracted from each page
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                cleaned_rows = []
                for row in table:
                    if not row: continue
                    # Convert None to empty string
                    cleaned_row = [cell if cell is not None else "" for cell in row]
                    cleaned_rows.append(" | ".join(cleaned_row))

                table_text = "\n".join(cleaned_rows).strip()

                if table_text:
                    results.append({
                        "content": table_text,
                        "modality": "table",
                        "page": page_num
                    })
    return results


# extracting images from the pdf file with OCR
def extract_images_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    results = []    # add text extracted from each image
    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        page_text = page.get_text().lower()
        for img in image_list:
            xref = img[0]  # xref = internal object ID inside the PDF
            base_image = doc.extract_image(xref)   # get the actual stored raw data from xref
            image_bytes = base_image["image"]   # image with binary bytes
            image = Image.open(io.BytesIO(image_bytes))  # actual image
            ocr_text = pytesseract.image_to_string(image).strip()

            if not ocr_text: continue

            combined_text = (
                "[FIGURE OR CHART]\n"
                "Context from page:\n"
                f"{page_text[:1500]}\n\n"
                "Extracted figure text:\n"
                f"{ocr_text}"
            )
            results.append({
                "content": combined_text,
                "modality": "figure",
                "page": page_num
            })
    return results


def ingest():
    all_chunks = []

    print("Extracting text...")
    all_chunks.extend(extract_text(PDF_PATH))

    print("Extracting tables...")
    all_chunks.extend(extract_tables(PDF_PATH))

    print("Extracting images...")
    all_chunks.extend(extract_images_with_ocr(PDF_PATH))

    # store all the chunks in chunks.json file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Ingestion complete. Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    ingest()

