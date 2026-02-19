"""
collect_imgs.py — Hand Sign Dataset Collector
Collects images for each alphabet (A-Z) using your webcam.
Enhanced: live preview, progress bar, auto-directory creation, flip support.
"""

import os
import cv2
import time
import argparse

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR        = "./data"
CLASSES         = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # A–Z
DATASET_SIZE    = 200       # images per class
COUNTDOWN_SECS  = 3         # seconds before capture starts
FLIP_FRAME      = True      # mirror the feed (selfie-cam feel)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Collect hand-sign images")
parser.add_argument("--size",    type=int, default=DATASET_SIZE, help="Images per class")
parser.add_argument("--classes", nargs="+", default=CLASSES,     help="Which classes to collect")
parser.add_argument("--camera",  type=int, default=0,            help="Camera index")
args = parser.parse_args()

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("[ERROR] Cannot open camera. Check --camera index.")
    exit(1)

print("\n🤚 Hand Sign Dataset Collector")
print(f"   Classes : {args.classes}")
print(f"   Per class: {args.size} images")
print("   Press  Q  at any time to quit.\n")

for label in args.classes:
    class_dir = os.path.join(DATA_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    # ── Waiting screen ────────────────────────────────────────────────────
    print(f"[{label}] Get ready — press SPACE to start collecting…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Class: {label}  |  Press SPACE to start",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)

        cv2.imshow("Hand Sign Collector", frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("\n[INFO] Collection stopped by user.")
            exit(0)

    # ── Countdown ─────────────────────────────────────────────────────────
    start_time = time.time()
    while time.time() - start_time < COUNTDOWN_SECS:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        remaining = int(COUNTDOWN_SECS - (time.time() - start_time)) + 1
        cv2.putText(frame, str(remaining),
                    (frame.shape[1] // 2 - 40, frame.shape[0] // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
        cv2.imshow("Hand Sign Collector", frame)
        cv2.waitKey(25)

    # ── Capture loop ──────────────────────────────────────────────────────
    counter = 0
    while counter < args.size:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        # Progress overlay
        progress = counter / args.size
        bar_w    = int(frame.shape[1] * progress)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (0, 0), (bar_w, 8), (0, 220, 100), -1)
        cv2.putText(frame, f"Class [{label}]  {counter}/{args.size}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Hand Sign Collector", frame)
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print(f"   ✓  {counter} images saved for class '{label}'")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Dataset collection complete! Run  create_dataset.py  next.\n")
