
# import sys, torch, torch.nn as nn, torch.optim as optim, numpy as np, random, time
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path

# # 1. SETUP
# BASE_DIR = Path(__file__).resolve().parent.parent 
# DATA_PATH = BASE_DIR / "data/processed/spectrograms"
# MODEL_SAVE_PATH = BASE_DIR / "models_saved/best_genre_model.pth"

# try:
#     from model_architecture4 import GenreCNN
# except ImportError:
#     print("Architecture file missing!"); sys.exit(1)

# # 2. DATASET (Softened Noise)
# class SpectrogramDataset(Dataset):
#     def __init__(self, root_dir, target_width=1292, augment=False):
#         self.root_dir = Path(root_dir)
#         self.file_list = list(self.root_dir.glob("**/*.npy"))
#         self.genres = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
#         self.label_map = {g: i for i, g in enumerate(self.genres)}
#         self.target_width, self.augment = target_width, augment

#     def __len__(self): return len(self.file_list)

#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         spec = np.load(file_path)
#         # Pad/Crop
#         if spec.shape[1] < self.target_width:
#             spec = np.pad(spec, ((0, 0), (0, self.target_width - spec.shape[1])), mode='constant', constant_values=-80)
#         else: spec = spec[:, :self.target_width]
#         # Augment
#         if self.augment:
#             if random.random() > 0.5: # Masking
#                 f = random.randint(0, 12); f0 = random.randint(0, 128-f); spec[f0:f0+f, :] = -80
#             if random.random() > 0.5: # Softened Noise (0.2)
#                 spec = spec + np.random.normal(0, 0.2, spec.shape)
#         return torch.tensor((spec+80)/80, dtype=torch.float32), torch.tensor(self.label_map[file_path.parent.name], dtype=torch.long)

# # 3. MAIN TRAINING
# if __name__ == "__main__":
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     full_ds = SpectrogramDataset(DATA_PATH)
#     tr_idx, val_idx = torch.utils.data.random_split(range(len(full_ds)), [800, 200], generator=torch.Generator().manual_seed(42))
    
#     train_loader = DataLoader(SpectrogramDataset(DATA_PATH, augment=True), batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(tr_idx))
#     val_loader = DataLoader(SpectrogramDataset(DATA_PATH, augment=False), batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

#     model = GenreCNN().to(DEVICE)
#     # Balanced Weights
#     weights = torch.tensor([1.1, 1.0, 1.2, 1.2, 1.1, 1.0, 1.0, 1.5, 1.1, 1.6]).to(DEVICE)
#     criterion = nn.CrossEntropyLoss(weight=weights)
#     optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

#     print(f"🚀 FINAL LAUNCH on: {DEVICE}")
#     total_start, best_acc, patience, counter = time.time(), 0.0, 7, 0

#     for epoch in range(1, 41):
#         epoch_start = time.time()
#         model.train()
#         l_sum = 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad(); loss = criterion(model(inputs), labels); loss.backward(); optimizer.step()
#             l_sum += loss.item()

#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#                 _, pred = torch.max(model(inputs), 1)
#                 total += labels.size(0); correct += (pred == labels).sum().item()

#         acc = 100 * correct / total
#         scheduler.step(); epoch_dur = time.time() - epoch_start
#         print(f"Epoch [{epoch}/40] - Val Acc: {acc:.2f}% - Time: {epoch_dur:.1f}s")

#         if acc > best_acc:
#             best_acc = acc; torch.save(model.state_dict(), MODEL_SAVE_PATH)
#             print(f"  ⭐ Saved Best: {best_acc:.2f}%"); counter = 0
#         elif counter >= patience: print("Early stop!"); break
#         else: counter += 1

#     print(f"\n✅ Done! Total Time: {(time.time()-total_start)/60:.2f} mins")













import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

# Import model architecture
try:
    from model_architecture4 import GenreCNN
except ImportError:
    print("Architecture file missing.")
    sys.exit(1)


# -----------------------------
# 1. PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "spectrograms"
MODEL_SAVE_PATH = BASE_DIR / "models_saved" / "best_genre_model.pth"


# -----------------------------
# 2. DATASET
# -----------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, target_width=1292, augment=False):
        self.root_dir = Path(root_dir)
        self.file_list = list(self.root_dir.glob("**/*.npy"))
        self.genres = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.label_map = {genre: idx for idx, genre in enumerate(self.genres)}
        self.target_width = target_width
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        spec = np.load(file_path)

        # Pad or crop to fixed width for batching
        if spec.shape[1] < self.target_width:
            pad_amount = self.target_width - spec.shape[1]
            spec = np.pad(
                spec,
                ((0, 0), (0, pad_amount)),
                mode="constant",
                constant_values=-80
            )
        else:
            spec = spec[:, :self.target_width]

        # Optional augmentation during training
        if self.augment:
            # Random frequency masking
            if random.random() > 0.5:
                mask_height = random.randint(0, 12)
                start_bin = random.randint(0, 128 - mask_height)
                spec[start_bin:start_bin + mask_height, :] = -80

            # Small Gaussian noise
            if random.random() > 0.5:
                spec = spec + np.random.normal(0, 0.2, spec.shape)

        # Normalize dB spectrogram to [0, 1]
        spec_norm = (spec + 80) / 80

        label = self.label_map[file_path.parent.name]

        return (
            torch.tensor(spec_norm, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# -----------------------------
# 3. TRAINING
# -----------------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = SpectrogramDataset(DATA_PATH, augment=False)

    # Safer percentage-based split instead of fixed [800, 200]
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_subset, val_subset = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        SpectrogramDataset(DATA_PATH, augment=True),
        batch_size=16,
        sampler=SubsetRandomSampler(train_subset.indices)
    )

    val_loader = DataLoader(
        SpectrogramDataset(DATA_PATH, augment=False),
        batch_size=16,
        sampler=SubsetRandomSampler(val_subset.indices)
    )

    model = GenreCNN().to(DEVICE)

    # Class weights help if some genres are harder or slightly imbalanced
    class_weights = torch.tensor(
        [1.1, 1.0, 1.2, 1.2, 1.1, 1.0, 1.0, 1.5, 1.1, 1.6],
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    print(f"Training on: {DEVICE}")

    total_start = time.time()
    best_acc = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(1, 41):
        epoch_start = time.time()

        # ---- Training ----
        model.train()
        train_loss_sum = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        # ---- Validation ----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100.0 * correct / total
        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_train_loss = train_loss_sum / len(train_loader)

        print(
            f"Epoch [{epoch}/40] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Saved best model: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    total_minutes = (time.time() - total_start) / 60
    print(f"\nDone. Total training time: {total_minutes:.2f} minutes")