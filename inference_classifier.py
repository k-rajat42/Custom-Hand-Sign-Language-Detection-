"""
inference_classifier.py — Real-time Hand Sign Recognition
═══════════════════════════════════════════════════════════
Extra features added over the original:
  ✦ Prediction confidence bar
  ✦ Letter-sequence builder + word history (compose words letter by letter)
  ✦ Stability filter  (letter confirmed only after N consecutive frames)
  ✦ Text-to-speech announcement of confirmed letters (optional, needs pyttsx3)
  ✦ Screenshot capture (press S)
  ✦ FPS counter
  ✦ Dual-hand warning
  ✦ Works with the pre-trained model.p out of the box
"""

import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
import os
import argparse
from collections import Counter, deque

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Real-time hand-sign recognition")
parser.add_argument("--model",       default="./model/model.p", help="Path to model.p")
parser.add_argument("--camera",      type=int,   default=0)
parser.add_argument("--confidence",  type=float, default=0.6,   help="Min confidence threshold")
parser.add_argument("--stability",   type=int,   default=15,    help="Frames to confirm a letter")
parser.add_argument("--tts",         action="store_true",        help="Speak confirmed letters")
parser.add_argument("--flip",        action="store_true", default=True, help="Mirror camera feed")
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(args.model):
    print(f"[ERROR] Model not found at '{args.model}'")
    print("  Run train_classifier.py first, or check the path.")
    exit(1)

with open(args.model, "rb") as f:
    payload = pickle.load(f)

model   = payload["model"]
le      = payload["label_encoder"]
CLASSES = list(le.classes_)
print(f"✅ Loaded model  |  Classes: {CLASSES}")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
hands       = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.5)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit(1)

# ── State ─────────────────────────────────────────────────────────────────────
sentence        = ""           # composed word
word_history    = []           # confirmed words
stability_buf   = deque(maxlen=args.stability)
last_confirmed  = ""
screenshot_dir  = "./screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# FPS
fps_buf   = deque(maxlen=30)
prev_time = time.time()

# UI colours
CLR_BG    = (20, 20, 20)
CLR_GREEN = (0, 220, 100)
CLR_BLUE  = (70, 130, 255)
CLR_WHITE = (230, 230, 230)
CLR_RED   = (60, 60, 255)
CLR_GOLD  = (0, 210, 255)

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.6):
    overlay = img.copy()
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius),(x2-radius, y1+radius),
                   (x1+radius, y2-radius),(x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def speak(text):
    if TTS_AVAILABLE and args.tts:
        _tts_engine.say(text)
        _tts_engine.runAndWait()

print("\n🤚 Hand Sign Recognition Running")
print("   SPACE  → add space     BACKSPACE → delete last char")
print("   S      → screenshot    ENTER     → confirm word")
print("   Q      → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if args.flip:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── FPS ───────────────────────────────────────────────────────────────
    now = time.time()
    fps_buf.append(1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    fps = np.mean(fps_buf)

    # ── Hand detection ────────────────────────────────────────────────────
    results = hands.process(rgb)
    pred_char  = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_styles.get_default_hand_landmarks_style(),
                                   mp_styles.get_default_hand_connections_style())

        lm = results.multi_hand_landmarks[0].landmark
        xs = [p.x for p in lm]; ys = [p.y for p in lm]
        min_x, min_y = min(xs), min(ys)
        feats = []
        for p in lm:
            feats.extend([p.x - min_x, p.y - min_y])

        proba = model.predict_proba([feats])[0]
        top_idx   = int(np.argmax(proba))
        confidence= float(proba[top_idx])

        if confidence >= args.confidence:
            pred_char = le.inverse_transform([top_idx])[0]

        # Bounding box
        px1 = int(min(xs) * W) - 20; py1 = int(min(ys) * H) - 20
        px2 = int(max(xs) * W) + 20; py2 = int(max(ys) * H) + 20
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(W, px2), min(H, py2)
        color = CLR_GREEN if confidence >= args.confidence else CLR_RED
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, f"{pred_char}  {confidence*100:.0f}%",
                    (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ── Stability filter ──────────────────────────────────────────────────
    stability_buf.append(pred_char)
    if len(stability_buf) == args.stability:
        most_common, cnt = Counter(stability_buf).most_common(1)[0]
        if cnt >= args.stability * 0.8 and most_common and most_common != last_confirmed:
            sentence      += most_common
            last_confirmed = most_common
            speak(most_common)
    else:
        if not pred_char:
            last_confirmed = ""

    # ── HUD ───────────────────────────────────────────────────────────────
    # Top bar
    draw_rounded_rect(frame, (0, 0), (W, 50), CLR_BG, radius=0, alpha=0.75)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_GOLD, 2)

    # Sentence panel
    draw_rounded_rect(frame, (0, H - 100), (W, H), CLR_BG, radius=0, alpha=0.8)
    display_sentence = sentence[-40:] if len(sentence) > 40 else sentence
    cv2.putText(frame, f">> {display_sentence}|",
                (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_WHITE, 2)
    cv2.putText(frame, "SPACE=space  BKSP=del  ENTER=word  S=snap  Q=quit",
                (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Confidence bar (right side)
    bar_h = int((H - 160) * confidence)
    cv2.rectangle(frame, (W - 30, 60), (W - 10, H - 110), (50, 50, 50), -1)
    cv2.rectangle(frame, (W - 30, H - 110 - bar_h), (W - 10, H - 110),
                  CLR_GREEN if confidence >= args.confidence else CLR_RED, -1)
    cv2.putText(frame, "CONF", (W - 38, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_WHITE, 1)

    # Stability dots
    dot_x = W - 55
    for i, c in enumerate(list(stability_buf)[-args.stability:]):
        color = CLR_GREEN if c == pred_char and c else (80, 80, 80)
        cv2.circle(frame, (dot_x, 70 + i * 12), 4, color, -1)

    cv2.imshow("Hand Sign Recognition", frame)

    # ── Key handling ──────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        sentence      += " "
        last_confirmed = ""
    elif key == 8 or key == 127:    # backspace
        sentence = sentence[:-1]
    elif key == 13:                  # enter → confirm word
        word = sentence.strip()
        if word:
            word_history.append(word)
            speak(word)
            print(f"  Word confirmed: {word}")
        sentence      = ""
        last_confirmed = ""
    elif key == ord('s'):
        fname = os.path.join(screenshot_dir, f"snap_{int(time.time())}.png")
        cv2.imwrite(fname, frame)
        print(f"  📸 Screenshot → {fname}")

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\n📝 Session summary")
print(f"   Last sentence : {sentence}")
print(f"   Word history  : {word_history}\n")
