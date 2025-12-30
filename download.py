from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_16_1", "fa", split="test")

import os
import shutil
import csv

save_dir = "path_to_data"
audio_dir = os.path.join(save_dir, "wavs")
os.makedirs(audio_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "transcriptions.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "transcription"])

    for i, example in enumerate(dataset):
        audio_info = example["audio"]
        original_audio_path = audio_info["path"]

        new_filename = f"audio_{i:06}.wav"
        new_audio_path = os.path.join(audio_dir, new_filename)

        shutil.copy(original_audio_path, new_audio_path)

        writer.writerow([new_filename, example["sentence"]])
