import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1 – Load PDFs with metadata
def load_pdfs(pdf_folder):
    docs = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    docs.append({
                        "text": text,
                        "metadata": {"source": filename, "page": page_num + 1}
                    })
    return docs

# Step 2 – Chunk text
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            chunked_docs.append((chunk, doc["metadata"]))
    return chunked_docs

# Step 3 – Create FAISS store
def create_vector_store(chunked_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [c[0] for c in chunked_docs]
    metadatas = [c[1] for c in chunked_docs]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    pdf_folder = "documents"
    print("📄 Reading PDFs...")
    docs = load_pdfs(pdf_folder)

    print("✂️ Splitting into chunks with metadata...")
    chunked_docs = chunk_documents(docs)

    print("🔢 Creating embeddings & saving FAISS index...")
    create_vector_store(chunked_docs)

    print("✅ FAISS index ready in 'faiss_index'")
