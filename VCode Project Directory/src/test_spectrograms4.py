# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Path Logic
# BASE_DIR = Path(__file__).resolve().parent.parent
# SPECTRO_DIR = BASE_DIR / "data/processed/spectrograms"

# def verify_and_plot():
#     # Find the first .npy file in the first genre folder it finds
#     npy_files = list(SPECTRO_DIR.glob("**/*.npy"))
    
#     if not npy_files:
#         print("No .npy files found! Check your data/processed/spectrograms folder.")
#         return

#     sample_path = npy_files[0]
#     print(f"Loading: {sample_path}")
    
#     # Load the array
#     mels_db = np.load(sample_path)
    
#     # Check the "Health" of the data
#     print("--- Data Stats ---")
#     print(f"Shape: {mels_db.shape}") # Looking for (128, ~1290-1300)
#     print(f"Min dB: {mels_db.min():.2f}")
#     print(f"Max dB: {mels_db.max():.2f}")
#     print("------------------")

#     # Plot it
#     plt.figure(figsize=(10, 4))
#     # Note: We use fmax=8000 because that's what we set in the preprocessing
#     librosa.display.specshow(mels_db, x_axis='time', y_axis='mel', sr=22050, fmax=8000)
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(f"Spectrogram Check: {sample_path.name}")
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     verify_and_plot()








import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SPECTRO_DIR = BASE_DIR / "data" / "processed" / "spectrograms"


def verify_and_plot():
    # Find any saved spectrogram file
    npy_files = list(SPECTRO_DIR.glob("**/*.npy"))

    if not npy_files:
        print("No .npy files found. Check your processed spectrogram folder.")
        return

    sample_path = npy_files[0]
    print(f"Loading: {sample_path}")

    # Load stored Mel spectrogram in dB
    mel_db = np.load(sample_path)

    # Print quick health check stats
    print("\n--- Data Stats ---")
    print(f"Shape: {mel_db.shape}")
    print(f"Min dB: {mel_db.min():.2f}")
    print(f"Max dB: {mel_db.max():.2f}")
    print("------------------\n")

    # Plot the spectrogram for visual inspection
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=22050,
        fmax=8000
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram Check: {sample_path.name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_and_plot()