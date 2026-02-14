import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# ----------------------------
# Setup
# ----------------------------

app = FastAPI()
api=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api)

# ----------------------------
# Dummy Data (65 Reviews)
# ----------------------------

documents = [
    {"id": i, "content": text}
    for i, text in enumerate([
        "Battery life is terrible and drains quickly.",
        "Excellent camera quality and sharp display.",
        "The phone overheats during gaming.",
        "Fast shipping and good packaging.",
        "Customer service was very helpful.",
        "The battery lasts all day with heavy use.",
        "Sound quality is amazing.",
        "Screen cracked after one drop.",
        "Very lightweight and portable.",
        "Performance is slow after update.",
        "Great value for the price.",
        "Battery drains in 3 hours.",
        "Amazing build quality.",
        "Terrible customer support experience.",
        "Charging is extremely fast.",
        "The speakers are very loud.",
        "App crashes frequently.",
        "Comfortable to hold.",
        "Poor battery optimization.",
        "Fantastic display brightness.",
        "Very durable design.",
        "Touchscreen is unresponsive sometimes.",
        "Battery life improved after patch.",
        "Camera struggles in low light.",
        "Shipping was delayed.",
        "Very satisfied with performance.",
        "Battery replacement was easy.",
        "Overheats while charging.",
        "Solid battery backup.",
        "Bluetooth connectivity issues.",
        "Very smooth user interface.",
        "Battery died within months.",
        "Great battery life overall.",
        "Affordable and reliable.",
        "Performance lags occasionally.",
        "Battery lasts more than expected.",
        "Excellent battery efficiency.",
        "Phone shuts down randomly.",
        "Battery percentage inaccurate.",
        "Fantastic sound clarity.",
        "Battery swelling issue.",
        "Stylish design and color.",
        "Battery charges quickly.",
        "Speaker distortion at high volume.",
        "Battery health degraded fast.",
        "Amazing battery endurance.",
        "Phone freezes often.",
        "Battery backup exceeded expectations.",
        "High performance chipset.",
        "Battery issue resolved after update.",
        "Premium feel.",
        "Battery overheating problem.",
        "Crystal clear display.",
        "Battery lasts 2 days.",
        "Laggy performance.",
        "Battery consumption too high.",
        "Fast processor.",
        "Battery drains overnight.",
        "Solid construction.",
        "Battery performance disappointing.",
        "Smooth animations.",
        "Battery life decent.",
        "Great audio experience.",
        "Battery issue fixed with firmware."
    ])
]

# ----------------------------
# Precompute Embeddings
# ----------------------------

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

for doc in documents:
    doc["embedding"] = embed(doc["content"])

# ----------------------------
# Cosine Similarity
# ----------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Re-ranking (LLM-style scoring)
# ----------------------------

def rerank(query, candidates):
    results = []
    for doc, score in candidates:
        prompt = f"""
Query: "{query}"
Document: "{doc['content']}"
Rate relevance 0-10.
Respond with only the number.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_score = float(response.choices[0].message.content.strip())
        normalized = raw_score / 10
        results.append((doc, normalized))
    return results

# ----------------------------
# API Model
# ----------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6

# ----------------------------
# Search Endpoint
# ----------------------------

@app.post("/search")
def search(req: SearchRequest):

    start = time.time()

    query_embedding = embed(req.query)

    # Initial retrieval
    similarities = [
        (doc, cosine_similarity(query_embedding, doc["embedding"]))
        for doc in documents
    ]

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:req.k]

    # Re-ranking
    if req.rerank:
        reranked = rerank(req.query, top_k)
        reranked.sort(key=lambda x: x[1], reverse=True)
        final = reranked[:req.rerankK]
    else:
        final = top_k[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": [
            {
                "id": doc["id"],
                "score": round(score, 4),
                "content": doc["content"],
                "metadata": {"source": "review"}
            }
            for doc, score in final
        ],
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
