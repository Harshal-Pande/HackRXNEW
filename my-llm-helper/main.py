from PyPDF2 import PdfReader
import os

# Folder where PDFs are stored
pdf_folder = "documents"

# Loop through all PDF files in folder
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        reader = PdfReader(pdf_path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        print(f"---- {filename} ----")
        print(text[:500])  # Show first 500 chars from this file
        print("\n")
