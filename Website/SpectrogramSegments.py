import os
import librosa
import numpy as np

dataset_path = r"C:\Users\Owner\Documents\Year4Classes\Spring\ECE_AI\Project\Dataset\genres"
output_path = r"C:\Users\Owner\Documents\Year4Classes\Spring\ECE_AI\Project\segments"

os.makedirs(output_path, exist_ok=True)

segment_length = 3  # seconds

print("Generating segmented spectrogram dataset...")

for genre in os.listdir(dataset_path):

    genre_path = os.path.join(dataset_path, genre)

    if not os.path.isdir(genre_path):
        continue

    print("Processing genre:", genre)

    genre_output = os.path.join(output_path, genre)
    os.makedirs(genre_output, exist_ok=True)

    for file in os.listdir(genre_path):

        if not file.endswith(".au"):
            continue

        file_path = os.path.join(genre_path, file)

        try:
            # Load full song
            y, sr = librosa.load(file_path)

            samples_per_segment = segment_length * sr
            num_segments = len(y) // samples_per_segment

            for i in range(num_segments):

                start = i * samples_per_segment
                end = start + samples_per_segment

                segment = y[start:end]

                # Create spectrogram
                spec = librosa.feature.melspectrogram(
                    y=segment,
                    sr=sr,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128
                )

                spec_db = librosa.power_to_db(spec, ref=np.max)

                save_name = file.replace(".au", "") + f"_seg{i}.npy"

                save_path = os.path.join(genre_output, save_name)

                np.save(save_path, spec_db)

        except Exception as e:
            print("Error processing:", file_path)
            print(e)

print("Finished generating segmented spectrogram dataset.")