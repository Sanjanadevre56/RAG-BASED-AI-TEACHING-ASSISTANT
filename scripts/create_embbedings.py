import requests
import os
import json
import pandas as pd
import joblib


# ---------------------------------
# CREATE EMBEDDING FUNCTION
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
# READ JSON FILES
# ---------------------------------
jsons = os.listdir("jsons")

my_dicts = []
chunk_id = 0

for json_file in jsons:

    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating embeddings for {json_file}")

    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):

        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]

        chunk_id += 1
        my_dicts.append(chunk)


# ---------------------------------
# CREATE DATAFRAME
# ---------------------------------
df = pd.DataFrame.from_records(my_dicts)

print("Total Chunks:", len(df))


# ---------------------------------
# SAVE EMBEDDINGS
# ---------------------------------
joblib.dump(df, "embeddings.joblib")

print("Embeddings saved successfully")