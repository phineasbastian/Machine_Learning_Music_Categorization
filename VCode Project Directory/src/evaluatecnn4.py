# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
# from torch.utils.data import DataLoader
# from pathlib import Path
# import sys

# # 1. PATH SETUP
# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_PATH = BASE_DIR / "data" / "processed" / "spectrograms"
# MODEL_PATH = BASE_DIR / "models_saved" / "best_genre_model.pth"

# # 2. IMPORT FROM TRAIN SCRIPT (Safety Gate in traincnn4 prevents re-training)
# try:
#     from traincnn4 import SpectrogramDataset
#     from model_architecture4 import GenreCNN
# except ImportError:
#     print("Error: Ensure traincnn4.py and model_architecture4.py are in the src folder.")
#     sys.exit(1)

# def evaluate():
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 3. LOAD VALIDATION DATA (Seed 42 matches the training split)
#     full_dataset = SpectrogramDataset(DATA_PATH)
#     _, val_db = torch.utils.data.random_split(
#         full_dataset, [800, 200], generator=torch.Generator().manual_seed(42)
#     )
#     val_loader = DataLoader(val_db, batch_size=1, shuffle=False)

#     # 4. LOAD THE SAVED MODEL
#     model = GenreCNN(num_classes=10).to(DEVICE)
#     if not MODEL_PATH.exists():
#         print(f"Error: Could not find model file at {MODEL_PATH}")
#         return

#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()
#     print("🚀 Model loaded. Analyzing music genres...")

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs = inputs.to(DEVICE)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.append(predicted.item())
#             all_labels.append(labels.item())

#     # 5. PLOT RESULTS
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(12, 9))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=full_dataset.genres, 
#                 yticklabels=full_dataset.genres)
#     plt.xlabel('Predicted Genre')
#     plt.ylabel('Actual Genre')
#     plt.title('Milestone 4: CNN Genre Classification Confusion Matrix')
    
#     plt.savefig(BASE_DIR / "confusion_matrix_cnn.png")
#     print("Confusion Matrix saved as 'confusion_matrix_cnn.png'")
#     plt.show()

#     print("\n--- Classification Report ---")
#     print(classification_report(all_labels, all_preds, target_names=full_dataset.genres))

# if __name__ == "__main__":
#     evaluate()








import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# 1. PATH SETUP
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "spectrograms"
MODEL_PATH = BASE_DIR / "models_saved" / "best_genre_model.pth"

try:
    from traincnn4 import SpectrogramDataset
    from model_architecture4 import GenreCNN
except ImportError:
    print("Error: Ensure scripts are in the src folder.")
    sys.exit(1)

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. LOAD DATA (Ensuring seed matches training split)
    full_dataset = SpectrogramDataset(DATA_PATH)
    _, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [800, 200], generator=torch.Generator().manual_seed(42)
    )
    val_db = SpectrogramDataset(DATA_PATH, augment=False)
    val_db.file_list = [full_dataset.file_list[i] for i in val_indices]
    val_loader = DataLoader(val_db, batch_size=1, shuffle=False)

    # 3. LOAD MODEL
    model = GenreCNN(num_classes=10).to(DEVICE)
    if not MODEL_PATH.exists():
        print(f"Error: No model found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("🚀 Running Final Evaluation...")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.item())
            all_labels.append(labels.item())

    # 4. PLOT CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_dataset.genres, yticklabels=full_dataset.genres)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Final Milestone 4 CNN Confusion Matrix')
    plt.tight_layout()
    plt.savefig(BASE_DIR / "final_confusion_matrix.png")
    plt.show()

    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.genres))

if __name__ == "__main__":
    evaluate()