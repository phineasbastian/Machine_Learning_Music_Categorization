import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path Logic
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data/raw"
OUT_DIR = BASE_DIR / "data/processed/spectrograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#converts the audio files to npy spectrograms
def create_spectrograms(src_path, dest_path):
    # Defining the 10 genres from the GTZAN dataset
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    
    for g in genres:
        print(f"Processing {g}...")
        genre_path = src_path / g
        save_path = dest_path / g
        save_path.mkdir(exist_ok=True)
        
        for filename in os.listdir(genre_path):
            if filename.endswith(".au"):
                try:
                    # Load audio (trimming to 30s to keep consistent shapes)
                    y, sr = librosa.load(genre_path / filename, duration=30.0)
                    
                    # Compute Mel-Spectrogram
                    # n_mels=128 is standard for high-res frequency bins
                    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                    mels_db = librosa.power_to_db(mels, ref=np.max)
                    
                    # Save as .npy for precision (better than JPG for AI artifact detection)
                    np.save(save_path / f"{filename[:-3]}.npy", mels_db)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    create_spectrograms(RAW_DATA, OUT_DIR)