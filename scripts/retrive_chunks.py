import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------
# CREATE EMBEDDING
# ---------------------------------
def create_embedding(text_list):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    embeddings = response.json()["embeddings"]
    return embeddings


# ---------------------------------
# LOAD EMBEDDINGS
# ---------------------------------
df = joblib.load("embeddings.joblib")

print("Embeddings Loaded:", df.shape)


# ---------------------------------
# ASK QUESTION
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


print("\nSimilarity Scores:")
print(similarities)


# ---------------------------------
# TOP RESULTS
# ---------------------------------
top_results = 10
top_indices = similarities.argsort()[::-1][:top_results]

print("\nTop indexes:", top_indices)


# ---------------------------------
# SHOW RESULTS
# ---------------------------------
results = df.iloc[top_indices]

print("\nMost Relevant Chunks:\n")

for _, row in results.iterrows():

    print("Lecture No:", row["lecture_no"])
    print("Start Time:", row["start_time"])
    print("End Time:", row["end_time"])
    print("Text:", row["text"])
    print("-" * 60)