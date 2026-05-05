import os
import librosa
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_PATH = "data/raw"
PROCESSED_CSV_PATH = "data/processed/csv"
LOG_FILE = "data/processed/data_integrity_log.txt"

# Setup logging to keep track of corrupted files
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def clean_and_split_gtzan():
    """
    Milestone 1: Cleans GTZAN, handles dead files, and performs stratified splitting.
    """
    valid_data = []
    corrupted_count = 0

    print("🚀 Starting Milestone 1: Data Preparation...")
    
    # 1. Verify Directory Structure
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: {RAW_DATA_PATH} not found. Check your folder structure.")
        return

    genres = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
    
    # 2. Integrity Check (Handling Dead Files like jazz.00054.au)
    print("🔎 Checking 1,000 files for integrity...")
    for genre in genres:
        genre_folder = os.path.join(RAW_DATA_PATH, genre)
        for filename in os.listdir(genre_folder):
            if filename.endswith(".au"):
                file_path = os.path.join(genre_folder, filename)
                try:
                    # Attempt to load a tiny slice to verify the header/codec
                    librosa.load(file_path, duration=0.1)
                    valid_data.append({'file_path': file_path, 'genre': genre})
                except Exception as e:
                    logging.info(f"CORRUPTED: {file_path} | Error: {str(e)}")
                    print(f" Skipping {filename} (Corrupted)")
                    corrupted_count += 1

    # Convert to DataFrame
    df = pd.DataFrame(valid_data)
    print(f"✅ Verification Complete. Valid: {len(df)} | Corrupted: {corrupted_count}")

    # 3. Stratified Splitting (70% Train, 15% Val, 15% Test)
    print("📊 Performing Stratified Split...")
    
    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['genre']
    )

    # Second split: Split the 30% Temp into exactly 50/50 (15% each of total)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['genre']
    )

    # 4. Save splits to CSV
    os.makedirs(PROCESSED_CSV_PATH, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_CSV_PATH, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_CSV_PATH, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_CSV_PATH, "test.csv"), index=False)

    print(f"💾 Milestone 1 Complete. CSVs saved to {PROCESSED_CSV_PATH}")
    print(f"Final Counts -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    clean_and_split_gtzan()
    
    
    
    
    