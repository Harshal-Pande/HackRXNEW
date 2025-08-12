from fastapi import FastAPI
from pydantic import BaseModel
from query import chain, vectorstore  # Import from your existing query.py
import json

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Insurance Claim Decision API is running!"}

@app.post("/query")
def query_insurance(data: QueryRequest):
    docs = vectorstore.similarity_search(data.question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    raw_answer = chain.run(question=data.question, context=context)

    try:
        answer = json.loads(raw_answer)
    except:
        answer = {"decision": "Unable to parse", "raw_output": raw_answer}

    sources = list(set([f"{d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', '?')})" for d in docs]))
    answer["sources"] = sources
    return answer
