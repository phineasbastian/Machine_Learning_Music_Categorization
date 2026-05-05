import static_ffmpeg
# --- 1. THE CRITICAL MAC FIX ---
static_ffmpeg.add_paths() 

import os, torch, torch.nn as nn, librosa, librosa.display, numpy as np, gradio as gr, matplotlib.pyplot as plt, uuid
from pathlib import Path

# --- 2. SYSTEM SETUP ---
plt.switch_backend('Agg') 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models_saved" / "Detective_Final_Master.pth"
CLASSES = ["Classical/Acoustic", "Hip-Hop/Urban", "Rock/Aggressive", "Pop/Electronic"]

# --- 3. ARCHITECTURE ---
class DetectiveCNN(nn.Module):
    def __init__(self):
        super(DetectiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2_freq = nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0))
        self.conv2_time = nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.genre_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 4))
        self.forensic_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1))

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x_f, x_t = self.conv2_freq(x), self.conv2_time(x)
        x = self.pool(torch.relu(self.bn2(torch.cat([x_f, x_t], dim=1))))
        if x.device.type == "mps":
            x = x.to("cpu")
            x = self.adaptive_pool(x).to(DEVICE)
        else: x = self.adaptive_pool(x)
        x_flat = self.flatten(x)
        return self.genre_head(x_flat), torch.sigmoid(self.forensic_head(x_flat))

# --- 4. ENGINE INITIALIZATION ---
model = DetectiveCNN().to(DEVICE)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    # FIXED: Clean startup message
    print("✅ Forensic & Genre Multitask Engine: Master Model online.")

# --- 5. ANALYSIS ENGINE ---
def analyze(audio_path, sensitivity):
    if not audio_path: return None, None, "No file uploaded", None
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        window_size = sr * 30
        num_windows = max(1, len(y) // window_size)

        all_g_probs, all_ai_scores = [], []

        for i in range(num_windows):
            chunk = y[i * window_size : (i + 1) * window_size]
            if len(chunk) < window_size:
                chunk = librosa.util.fix_length(chunk, size=window_size)

            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = torch.tensor(
                librosa.util.fix_length(np.clip((mel_db + 80) / 80, 0, 1), size=1280, axis=1)
            ).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

            with torch.no_grad():
                g_logits, f_prob = model(mel_norm)
                g_probs = torch.softmax(g_logits, dim=1).cpu().numpy()[0]
                all_g_probs.append(g_probs)
                all_ai_scores.append(f_prob.item())

        avg_g   = np.mean(all_g_probs, axis=0)
        avg_ai  = float(np.mean(all_ai_scores))

        threshold = 0.95 - (sensitivity / 100) * 0.90
        verdict = "AI GENERATED" if avg_ai > threshold else "HUMAN ORIGINAL"

        res_genre    = {CLASSES[i]: float(avg_g[i]) for i in range(4)}
        res_forensic = {"Human": 1.0 - avg_ai, "AI": avg_ai}

        mel_viz = librosa.feature.melspectrogram(y=y[:window_size], sr=sr, n_mels=128)
        mel_db_viz = librosa.power_to_db(mel_viz, ref=np.max)
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(mel_db_viz, sr=sr, cmap='magma')
        plt.axis('off')
        img_name = f"spec_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        return res_genre, res_forensic, verdict, img_name

    except Exception as e: return {}, {}, f"Error: {str(e)}", None

# --- 6. THE DASHBOARD ---
with gr.Blocks(title="AI Music Analyzer", theme="soft") as demo:
    gr.Markdown("# AI Music Analyzer")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="Upload Track")
            slider = gr.Slider(0, 100, 50, label="Sensitivity")
            btn = gr.Button("Analyze Track", variant="primary")
        with gr.Column(scale=1):
            v_out = gr.Textbox(label="Final Verdict")
            g_out = gr.Label(label="Genre Classification")
            f_out = gr.Label(label="Creative Probability")
            img_out = gr.Image(label="Spectral Signature")

    btn.click(analyze, inputs=[audio_in, slider], outputs=[g_out, f_out, v_out, img_out])

# FIXED: Generates a clickable localhost (127.0.0.1) link
demo.launch(show_api=False)