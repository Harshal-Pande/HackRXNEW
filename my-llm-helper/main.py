import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1 – Load PDFs from the documents folder
def load_pdfs(pdf_folder):
    all_text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    return all_text

# Step 2 – Chunk the text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters in a chunk
        chunk_overlap=200 # Overlap for context
    )
    return splitter.split_text(text)

# Step 3 – Create embeddings and store in FAISS
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

if __name__ == "__main__":
    pdf_folder = "documents"
    print("📄 Reading PDFs...")
    text = load_pdfs(pdf_folder)

    print("✂️ Splitting into chunks...")
    chunks = chunk_text(text)

    print("🔢 Creating embeddings and saving vector store...")
    create_vector_store(chunks)

    print("✅ All done! Your FAISS index is ready in the 'faiss_index' folder.")
