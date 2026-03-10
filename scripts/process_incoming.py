import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


# ---------------------------------
# CREATE EMBEDDING
# ---------------------------------
def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    return r.json()["embeddings"]


# ---------------------------------
# LLM INFERENCE
# ---------------------------------
def inference(prompt):

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]


# ---------------------------------
# LOAD EMBEDDINGS
# ---------------------------------
print("Loading embeddings...")

df = joblib.load("embeddings.joblib")

print("Embeddings Loaded:", df.shape)


# ---------------------------------
# USER QUESTION
# ---------------------------------
incoming_query = input("\nAsk a Question: ")


# ---------------------------------
# CREATE QUESTION EMBEDDING
# ---------------------------------
question_embedding = create_embedding([incoming_query])[0]


# ---------------------------------
# COMPUTE SIMILARITY
# ---------------------------------
embedding_matrix = np.vstack(df["embedding"].values)

similarities = cosine_similarity(
    embedding_matrix,
    [question_embedding]
).flatten()


# ---------------------------------
# TOP RESULTS
# ---------------------------------
top_results = 5
top_indices = similarities.argsort()[::-1][:top_results]

new_df = df.iloc[top_indices]


# ---------------------------------
# CREATE CONTEXT
# ---------------------------------
context = new_df[["lecture_no","start_time","end_time","text"]].to_json(orient="records")


prompt = f"""
You are an AI teaching assistant.

Below are lecture subtitle chunks from a course.

Each chunk contains:
- lecture number
- start time (seconds)
- end time (seconds)
- spoken text

Lecture Chunks:
{context}

-------------------------------------

User Question:
{incoming_query}

Instructions:
- Tell the user WHICH lecture contains the answer
- Mention the exact timestamps
- Explain in simple language
- If the question is unrelated, say you can only answer questions related to the lectures
"""

# ---------------------------------
# GET AI RESPONSE
# ---------------------------------
response = inference(prompt)

print("\nAI ANSWER:\n")
print(response)