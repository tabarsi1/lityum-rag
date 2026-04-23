from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

documents = [
    "NevinDiri is a super engineer with expertise in machine learning and data science.",
    "Lityum Engineering is a leading manufacturer of precision CNC machines.",
    "Coolant pressure is maintained between 50-80 PSI.",
]

def embed(text):
    res = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(res.data[0].embedding)

doc_embeddings = [embed(doc) for doc in documents]

def retrieve(query, top_k=2):
    q_emb = embed(query)
    scores = [np.dot(q_emb, d) for d in doc_embeddings]
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in top_idx]

def ask(question):
    context = retrieve(question)
    context_text = "\n".join(context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
             "You are a manufacturing assistant. Answer using only "
             "the provided context. Be precise and concise."},
            {"role": "user", "content":
             f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

print(ask("Who is NevinDiri?"))
print(ask("What Lityum Engineering does?"))
print(ask("What is coolant pressure range?"))

