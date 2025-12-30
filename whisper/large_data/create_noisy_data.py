from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise
import os
import soundfile as sf
from glob import glob


# Augmentation pipeline
augment = Compose([
    AddBackgroundNoise(
        sounds_path="path_to_musan_noise",
        min_snr_db=5,
        max_snr_db=15,
        p=0.95
    ),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
])

# Load speech WAV files
speech_files = sorted(glob("path_to_wav_files"))

# Output folder
output_dir = "output_path"
os.makedirs(output_dir, exist_ok=True)

for i, wav_path in enumerate(speech_files):
    audio, sr = sf.read(wav_path)

    # Apply augmentation
    noisy_audio = augment(samples=audio, sample_rate=sr)

    output_path = os.path.join(output_dir, f"noisy_test_{i}.wav")
    sf.write(output_path, noisy_audio, samplerate=sr)

    if i % 10 == 0:
        print(f"Processed {i} files...")

print("All local WAV files processed.")
