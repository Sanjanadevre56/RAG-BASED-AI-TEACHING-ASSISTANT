import os
import json
import pandas as pd

# Folder containing transcript JSON files
TRANSCRIPT_FOLDER = "../jsons"

all_chunks = []

print("Reading transcript files...\n")

for file in os.listdir(TRANSCRIPT_FOLDER):

    if file.endswith(".json"):

        filepath = os.path.join(TRANSCRIPT_FOLDER, file)

        with open(filepath, "r", encoding="utf-8") as f:

            data = json.load(f)

        lecture_no = file.replace(".json", "")

        print("Processing:", lecture_no)

        for item in data["chunks"]:

            chunk = {
                "lecture_no": lecture_no,
                "start_time": item["start"],
                "end_time": item["end"],
                "text": item["text"]
            }

            all_chunks.append(chunk)


print("\nTotal chunks created:", len(all_chunks))


# Convert to DataFrame
df = pd.DataFrame(all_chunks)

# Save chunks
df.to_json("../chunks/output.json", orient="records")

print("\nChunks saved to chunks/output.json")