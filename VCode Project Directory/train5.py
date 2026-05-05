import sys, torch, torch.nn as nn, torch.optim as optim, numpy as np, random, time, os
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

# --- 1. MAC GPU CONFIG ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 NATIVE MAC GPU (MPS) DETECTED. Engaging M5 Neural Accelerators.")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  MPS not found. Using CPU.")

# --- 2. PATHS & CONFIG ---
BASE_DIR        = Path(__file__).resolve().parent
HUMAN_PATH      = BASE_DIR / "data" / "raw"
AI_PATH         = BASE_DIR / "data" / "Ai Audio"
MODEL_SAVE_PATH = BASE_DIR / "models_saved" / "Detective_Final_Master.pth"

SR           = 22050
DURATION     = 30
TARGET_WIDTH = 1280

os.makedirs(BASE_DIR / "models_saved", exist_ok=True)

SUPER_CLASS_MAP = {
    'classical': 0, 'jazz': 0,
    'hiphop':   1, 'reggae': 1,
    'metal':    2, 'rock':   2,
    'pop':      3, 'disco':  3, 'country': 3, 'blues': 3,
}

# --- 3. ARCHITECTURE ---
class DetectiveCNN(nn.Module):
    def __init__(self):
        super(DetectiveCNN, self).__init__()
        self.conv1       = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1         = nn.BatchNorm2d(32)
        self.conv2_freq  = nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0))
        self.conv2_time  = nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3))
        self.bn2         = nn.BatchNorm2d(128)
        self.pool        = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten     = nn.Flatten()
        self.genre_head  = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.ReLU(), nn.Linear(256, 4)
        )
        self.forensic_head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(
            torch.cat([self.conv2_freq(x), self.conv2_time(x)], dim=1)
        )))
        if x.device.type == "mps":
            x = self.adaptive_pool(x.to("cpu")).to(DEVICE)
        else:
            x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.genre_head(x), torch.sigmoid(self.forensic_head(x))

# --- 4. DATASET (raw audio → identical preprocessing to inference) ---
def audio_to_tensor(path):
    y, _ = librosa.load(str(path), sr=SR, duration=DURATION)
    y    = librosa.util.fix_length(y, size=SR * DURATION)
    mel  = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    db   = librosa.power_to_db(mel, ref=np.max)
    norm = np.clip((db + 80) / 80, 0, 1)
    norm = librosa.util.fix_length(norm, size=TARGET_WIDTH, axis=1)
    if random.random() > 0.5:
        norm = np.clip(norm + np.random.normal(0, 0.02, norm.shape), 0, 1)
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0)

AUDIO_EXTS = {'.wav', '.mp3', '.au', '.flac', '.ogg'}

class RawAudioDataset(Dataset):
    def __init__(self, human_dir, ai_dir):
        self.human_files = [f for f in Path(human_dir).glob("**/*") if f.suffix.lower() in AUDIO_EXTS]
        self.ai_files    = [f for f in Path(ai_dir).glob("**/*")    if f.suffix.lower() in AUDIO_EXTS]
        self.file_list   = self.human_files + self.ai_files
        self.labels      = [0.0] * len(self.human_files) + [1.0] * len(self.ai_files)
        print(f"✅ Found {len(self.human_files)} Human files.")
        print(f"🤖 Found {len(self.ai_files)} AI files.")

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        path     = self.file_list[idx]
        is_ai    = self.labels[idx]
        genre_idx = SUPER_CLASS_MAP.get(path.parent.name.lower(), 3)
        try:
            spec = audio_to_tensor(path)
        except Exception:
            spec = torch.zeros(1, 128, TARGET_WIDTH)
        return spec, torch.tensor(genre_idx, dtype=torch.long), torch.tensor(is_ai, dtype=torch.float32)

# --- 5. TRAINING ---
if __name__ == "__main__":
    full_ds = RawAudioDataset(HUMAN_PATH, AI_PATH)

    if len(full_ds) == 0:
        print(f"❌ ERROR: No files found!")
        sys.exit()

    weights = [1.0 / len(full_ds.human_files)] * len(full_ds.human_files) + \
              [1.0 / len(full_ds.ai_files)]    * len(full_ds.ai_files)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(full_ds, batch_size=32, sampler=sampler, num_workers=0)

    model          = DetectiveCNN().to(DEVICE)
    criterion_genre = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_ai    = nn.BCELoss()
    optimizer       = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)

    print(f"🔥 Starting Training — raw audio pipeline (matches inference exactly)")

    for epoch in range(1, 51):
        start = time.time()
        model.train()
        g_acc, ai_acc, total = 0, 0, 0

        for inputs, g_labels, ai_labels in train_loader:
            inputs, g_labels, ai_labels = inputs.to(DEVICE), g_labels.to(DEVICE), ai_labels.to(DEVICE)
            optimizer.zero_grad()
            g_out, ai_out = model(inputs)
            loss = (0.6 * criterion_genre(g_out, g_labels)) + \
                   (0.4 * criterion_ai(ai_out.squeeze(), ai_labels))
            loss.backward()
            optimizer.step()

            _, g_pred = torch.max(g_out, 1)
            g_acc  += (g_pred == g_labels).sum().item()
            ai_acc += ((ai_out.squeeze() > 0.5).float() == ai_labels).sum().item()
            total  += g_labels.size(0)

        print(f"Epoch {epoch}/50 | Genre: {100*g_acc/total:.1f}% | AI: {100*ai_acc/total:.1f}% | Time: {time.time()-start:.1f}s")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"✅ Master Model Saved to {MODEL_SAVE_PATH}")
