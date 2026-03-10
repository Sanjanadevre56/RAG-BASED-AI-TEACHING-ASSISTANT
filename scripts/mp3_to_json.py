import whisper
import json
import os

# Load Whisper model (better for CPU)
model = whisper.load_model("base")

audio_folder = "data/audios"
json_folder = "jsons"

os.makedirs(json_folder, exist_ok=True)

audios = os.listdir(audio_folder)

for audio in audios:

    if audio.endswith(".mp3"):

        # Remove .mp3
        filename = audio[:-4]

        parts = filename.split("_")

        number = parts[0]
        title = "_".join(parts[1:])

        print("Processing:", number, title)

        result = model.transcribe(
           audio=f"data/audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        chunks = []

        for segment in result["segments"]:

            chunks.append({
                "lecture_no": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        output = {
            "chunks": chunks,
            "text": result["text"]
        }

        with open(f"{json_folder}/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

print("\nAll audio files converted to JSON successfully.")