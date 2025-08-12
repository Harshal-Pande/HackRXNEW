from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json

# 1️⃣ Load FAISS index
print("📂 Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2️⃣ Load FLAN-T5-Large (CPU)
print("🧠 Loading local FLAN-T5-Large model on CPU...")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0,
    device=-1  # CPU mode
)

llm = HuggingFacePipeline(pipeline=pipe)

# 3️⃣ Improved Prompt
template = """
You are an expert insurance policy analyst.
Using the provided policy text, decide if the claim is approved.

If any part of the context suggests that the claim might be covered, lean towards approval unless there is explicit exclusion.

Query: {question}
Policy Context:
{context}

Return ONLY valid JSON:
{{
  "decision": "approved" or "rejected",
  "amount": number or null,
  "justification": "short explanation citing relevant clause"
}}
"""

prompt = PromptTemplate(input_variables=["question", "context"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# 4️⃣ Helper: keyword-boosted retrieval
def boosted_retrieval(query, k=8):
    docs = vectorstore.similarity_search(query, k=k)
    keywords = query.replace(",", " ").split()
    for kw in keywords:
        docs += vectorstore.similarity_search(kw, k=2)
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    return unique_docs

# 5️⃣ Interactive loop
while True:
    query = input("\n❓ Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    docs = boosted_retrieval(query, k=8)
    context = "\n\n".join([d.page_content for d in docs])

    raw_answer = chain.run(question=query, context=context)

    # Parse JSON
    try:
        answer = json.loads(raw_answer)
        print(f"\n📢 Decision: {answer['decision'].capitalize()}")
        if answer.get("amount"):
            print(f"💰 Amount: {answer['amount']}")
        print(f"📝 Justification: {answer['justification']}")
    except:
        print("\n⚠️ Could not parse as JSON. Raw Output:")
        print(raw_answer)

    # Show sources
    sources = list(set([f"{d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', '?')})" for d in docs]))
    print(f"📄 Covered in: {', '.join(sources) if sources else 'No source info'}")
