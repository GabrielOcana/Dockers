# cebra_train.py

import subprocess, sys

# Install dependencies
for package in [
    "git+https://github.com/AdaptiveMotorControlLab/cebra.git",
    "scikit-learn",
    "matplotlib"
]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import os, pickle, numpy as np, torch, cebra
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Prevent GUI issues on headless machines
plt.switch_backend("Agg")
plt.rcParams["figure.dpi"] = 110

# ---------------- 1. LOAD ---------------------------------------------------
data_path = os.path.join(os.environ["SM_CHANNEL_TRAIN"], "dataset.pkl")
with open(data_path, "rb") as f:
    data, labels = pickle.load(f)

# ---------------- 2. SUBSAMPLE ----------------------------------------------
rng         = np.random.default_rng(seed=42)
keep_idx    = rng.choice(len(data), size=len(data) // 4, replace=False)
data_half   = data[keep_idx, :, :10000]
labels_half = labels[keep_idx]

# ---------------- 3. PREP DATA FOR CEBRA ------------------------------------
sessions = [torch.from_numpy(chunk.T).float() for chunk in data_half]
index    = [torch.arange(s.shape[0]) for s in sessions]
y        = np.asarray(labels_half)

# ---------------- 4. DEVICE -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running CEBRA on:", device)

# ---------------- 5. TRAIN --------------------------------------------------
model = cebra.CEBRA(
    model_architecture="offset10-model",
    conditional="time",
    output_dimension=4,
    batch_size=50,
    max_iterations=10000,
    learning_rate=3e-4,
    temperature=0.15,
    device=device,
)
model.fit(sessions, index)

# ---------------- 6. EMBED & SAVE -------------------------------------------
Z_list     = model.transform(sessions)
Z_centroid = np.stack([z.mean(0).cpu().numpy() for z in Z_list])

clf  = LogisticRegression(max_iter=500).fit(Z_centroid, y)
auc  = roc_auc_score(y, clf.predict_proba(Z_centroid)[:, 1])
print(f"AUROC: {auc:.3f}")

# Save outputs to model directory
output_dir = os.environ["SM_MODEL_DIR"]
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "results.txt"), "w") as f:
    f.write(f"AUROC: {auc:.3f}\n")

plt.figure(figsize=(5, 5))
plt.scatter(Z_centroid[:, 0], Z_centroid[:, 1], s=12, c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("latent-0")
plt.ylabel("latent-1")
plt.title("CEBRA-Time centroids")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "embedding.png"))
