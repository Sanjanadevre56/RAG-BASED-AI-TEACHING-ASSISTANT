import whisper
import json

print("Loading model...")
model = whisper.load_model("base")

print("Starting transcription...")

result = model.transcribe(
    audio="audios/15773-sp24-lecture-1-version-3_360p_16_9.mp3",
    language="hi",
    task="translate",
    fp16=False
)

print("Transcription finished!")
print("Segments found:", len(result["segments"]))

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

chunks = []

for segment in result["segments"]:
    chunk_data = {
        "start_time": format_time(segment["start"]),
        "end_time": format_time(segment["end"]),
        "text": segment["text"].strip()
    }
    chunks.append(chunk_data)

print(json.dumps(chunks, indent=4))

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print("Saved to output.json")