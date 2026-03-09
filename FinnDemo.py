import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

spectrogram_path = r"C:\Users\Owner\Documents\Year4Classes\Spring\ECE_AI\Project\spectrograms"

genres = [g for g in os.listdir(spectrogram_path) if os.path.isdir(os.path.join(spectrogram_path, g))]

plt.figure(figsize=(15,10))

for i, genre in enumerate(genres):

    genre_folder = os.path.join(spectrogram_path, genre)
    files = [f for f in os.listdir(genre_folder) if f.endswith(".npy")]

    if len(files) == 0:
        continue

    spec = np.load(os.path.join(genre_folder, files[0]))

    plt.subplot(3,4,i+1)
    librosa.display.specshow(spec)
    plt.title(genre)
    plt.axis("off")

plt.tight_layout()
plt.show()