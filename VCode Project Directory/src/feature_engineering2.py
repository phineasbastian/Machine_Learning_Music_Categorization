import os
import numpy as np
import pandas as pd
import librosa
import warnings

# Suppress warnings from librosa regarding small files or specific audio formats
warnings.filterwarnings('ignore')

# PATH LOGIC:
# __file__ is .../src/feature_engineering.py
# 1st dirname is the 'src' folder
# 2nd dirname is the 'AI Music Genre Classification' root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV_DIR = os.path.join(BASE_DIR, "data", "processed", "csv")
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "data", "processed", "csv")

def extract_track_features(file_path):
    """
    Extracts signal processing features for a single audio file.
    Returns a dictionary of mean and variance for various audio metrics.
    """
    try:
        # Load audio (sr=22050 is standard for GTZAN)
        y, sr = librosa.load(file_path, duration=30)

        # --- 1. Time-Domain Features ---
        rms = librosa.feature.rms(y=y)
        # Amplitude Envelope
        ae = np.array([np.max(y[i:i+512]) for i in range(0, len(y), 512)])

        # --- 2. Frequency-Domain Features ---
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # --- 3. Perceptual Features ---
        # We extract 20 MFCCs to better catch AI-generated artifacts later
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Aggregation: Calculate Mean and Variance for every feature
        features = {}
        
        feature_sets = {
            "rms": rms,
            "amplitude_envelope": ae,
            "spectral_centroid": spec_centroid,
            "spectral_rolloff": spec_rolloff,
            "chroma": chroma
        }

        for name, data in feature_sets.items():
            features[f"{name}_mean"] = np.mean(data)
            features[f"{name}_var"] = np.var(data)

        for i in range(1, 21):
            features[f"mfcc{i}_mean"] = np.mean(mfcc[i-1])
            features[f"mfcc{i}_var"] = np.var(mfcc[i-1])

        return features

    except Exception as e:
        print(f"  [ERROR] Could not process {os.path.basename(file_path)}: {e}")
        return None

def main():
    splits = ['train.csv', 'val.csv', 'test.csv']
    
    print("--- Milestone 2: Feature Engineering Started ---")
    
    for split in splits:
        input_path = os.path.join(INPUT_CSV_DIR, split)
        
        if not os.path.exists(input_path):
            print(f"Skipping {split}: File not found at {input_path}")
            continue
            
        df = pd.read_csv(input_path)
        print(f"\nProcessing {split} ({len(df)} tracks)...")
        
        feature_data = []

        for index, row in df.iterrows():
            # Construct the absolute path to the audio file
            # Assuming row['file_path'] looks like 'data/raw/genre/filename.au'
            audio_path = os.path.join(BASE_DIR, row['file_path'])
            
            # Progress update every 50 files
            if index % 50 == 0:
                print(f"  Progress: {index}/{len(df)}")

            track_features = extract_track_features(audio_path)
            
            if track_features:
                track_features['genre'] = row['genre'] # Keep the target label
                feature_data.append(track_features)

        # Save the results
        output_df = pd.DataFrame(feature_data)
        output_filename = split.replace(".csv", "_features.csv")
        output_path = os.path.join(OUTPUT_CSV_DIR, output_filename)
        
        output_df.to_csv(output_path, index=False)
        print(f"Successfully saved: {output_filename}")

    print("\n--- Milestone 2 Complete ---")

if __name__ == "__main__":
    main()