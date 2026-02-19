"""
graph.py — Visualise dataset statistics and model performance.
Enhanced: class distribution, sample landmark plots, feature importance.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Visualise dataset & model stats")
parser.add_argument("--data",   default="./model/data.pickle")
parser.add_argument("--model",  default="./model/model.p")
parser.add_argument("--save",   default="./docs",  help="Where to save plots")
args = parser.parse_args()

os.makedirs(args.save, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(args.data, "rb") as f:
    ds = pickle.load(f)
X      = np.array(ds["data"])
labels = np.array(ds["labels"])

print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(set(labels))} classes")

# ── 1. Class distribution ─────────────────────────────────────────────────────
unique, counts = np.unique(labels, return_counts=True)
fig, ax = plt.subplots(figsize=(14, 4))
bars = ax.bar(unique, counts, color=sns.color_palette("viridis", len(unique)))
ax.set_title("Class Distribution (samples per letter)", fontsize=14, weight="bold")
ax.set_xlabel("Letter"); ax.set_ylabel("Count")
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, str(c),
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
path = os.path.join(args.save, "class_distribution.png")
plt.savefig(path, dpi=120); plt.close()
print(f"  ✓ {path}")

# ── 2. Sample landmark visualization (first 6 unique classes) ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
sample_classes = unique[:6]
for ax, cls in zip(axes.flat, sample_classes):
    idx   = np.where(labels == cls)[0][0]
    feat  = X[idx]           # 42 values: x0,y0, x1,y1, …
    xs    = feat[0::2]
    ys    = [-v for v in feat[1::2]]   # flip Y for natural display
    connections = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (0,9),(9,10),(10,11),(11,12),     # middle
        (0,13),(13,14),(14,15),(15,16),   # ring
        (0,17),(17,18),(18,19),(19,20),   # pinky
        (5,9),(9,13),(13,17),             # palm
    ]
    for a, b in connections:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "gray", lw=1.2)
    ax.scatter(xs, ys, c=range(21), cmap="plasma", s=60, zorder=5)
    ax.set_title(f"Letter  {cls}", fontsize=13, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Sample Hand Landmarks per Letter", fontsize=15, weight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(args.save, "sample_landmarks.png")
plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
print(f"  ✓ {path}")

# ── 3. Feature importance (if model exists) ───────────────────────────────────
if os.path.exists(args.model):
    with open(args.model, "rb") as f:
        payload = pickle.load(f)
    clf = payload["model"]
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        # Group into 21 landmarks
        landmark_importance = [fi[i*2] + fi[i*2+1] for i in range(21)]
        lm_names = [
            "Wrist","Thumb_CMC","Thumb_MCP","Thumb_IP","Thumb_Tip",
            "Idx_MCP","Idx_PIP","Idx_DIP","Idx_Tip",
            "Mid_MCP","Mid_PIP","Mid_DIP","Mid_Tip",
            "Ring_MCP","Ring_PIP","Ring_DIP","Ring_Tip",
            "Pink_MCP","Pink_PIP","Pink_DIP","Pink_Tip",
        ]
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = sns.color_palette("Spectral", 21)
        ax.bar(lm_names, landmark_importance, color=colors)
        ax.set_title("Feature Importance by Landmark (x+y combined)", fontsize=13, weight="bold")
        ax.set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        plt.tight_layout()
        path = os.path.join(args.save, "feature_importance.png")
        plt.savefig(path, dpi=120); plt.close()
        print(f"  ✓ {path}")

print(f"\n✅ All plots saved to  {args.save}/")
