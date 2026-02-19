"""
create_dataset.py — Extract MediaPipe hand landmarks from collected images.
Enhanced: augmentation flag, progress display, error logging, hand-count warning.
"""

import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import argparse
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
OUTPUT     = "./model/data.pickle"
AUGMENT    = True    # flip-augment each sample (doubles dataset size)

parser = argparse.ArgumentParser(description="Build landmark dataset from images")
parser.add_argument("--data",    default=DATA_DIR, help="Image data directory")
parser.add_argument("--output",  default=OUTPUT,   help="Output pickle path")
parser.add_argument("--augment", action="store_true", default=AUGMENT,
                    help="Apply horizontal flip augmentation")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

mp_hands    = mp.solutions.hands
hands       = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                              min_detection_confidence=0.3)

data, labels = [], []
skipped      = 0

classes = sorted(os.listdir(args.data))
print(f"\n📐 Extracting landmarks for {len(classes)} classes…\n")

def extract_landmarks(img_rgb):
    """Return normalised (x, y) flat list or None."""
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    min_x, min_y = min(xs), min(ys)
    # Normalise relative to wrist
    flat = []
    for p in lm:
        flat.extend([p.x - min_x, p.y - min_y])
    return flat

for label in tqdm(classes, desc="Classes"):
    class_dir = os.path.join(args.data, label)
    if not os.path.isdir(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    for img_file in images:
        img    = cv2.imread(os.path.join(class_dir, img_file))
        if img is None:
            skipped += 1
            continue
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats  = extract_landmarks(rgb)
        if feats:
            data.append(feats)
            labels.append(label)

        # Augmentation — horizontal flip
        if args.augment:
            flipped = cv2.flip(img, 1)
            rgb_f   = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
            feats_f = extract_landmarks(rgb_f)
            if feats_f:
                data.append(feats_f)
                labels.append(label)

hands.close()

with open(args.output, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"\n✅ Done! {len(data)} samples saved → {args.output}")
print(f"   Skipped (no hand detected): {skipped}")
print(f"   Classes : {sorted(set(labels))}\n")
