from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1️⃣ Load FAISS index
print("📂 Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2️⃣ Load FLAN-T5 model locally (CPU mode)
print("🧠 Loading local FLAN-T5 model on CPU...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0,
    device=-1  # -1 = CPU
)

# 3️⃣ Create LLM pipeline for LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# 4️⃣ Create RetrievalQA chain
print("🤖 Creating QA system...")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5️⃣ Interactive query loop
while True:
    query = input("\n❓ Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    result = qa(query)
    print("\n📢 Response:", result["result"])

    # Format sources neatly
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]))
    sources_str = ", ".join(sources)
    print(f"📄 Covered in: {sources_str if sources_str else 'No source info found'}")
