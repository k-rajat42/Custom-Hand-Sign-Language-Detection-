"""
train_classifier.py — Train a RandomForest model on hand-landmark features.
Enhanced: train/val/test split, cross-validation, confusion matrix, model export.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics            import (accuracy_score, classification_report,
                                        confusion_matrix)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train hand-sign classifier")
parser.add_argument("--data",       default="./model/data.pickle")
parser.add_argument("--output",     default="./model/model.p")
parser.add_argument("--model",      choices=["rf", "gb"], default="rf",
                    help="rf=RandomForest  gb=GradientBoosting")
parser.add_argument("--trees",      type=int, default=200,  help="n_estimators")
parser.add_argument("--test-size",  type=float, default=0.2)
parser.add_argument("--no-plot",    action="store_true")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(args.data, "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"\n🧠 Training Hand-Sign Classifier")
print(f"   Samples  : {len(X)}")
print(f"   Classes  : {list(le.classes_)}")
print(f"   Features : {X.shape[1]}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=args.test_size, stratify=y_enc, random_state=42
)

# ── Model ─────────────────────────────────────────────────────────────────────
if args.model == "rf":
    clf = RandomForestClassifier(
        n_estimators=args.trees, max_depth=None,
        min_samples_split=2, random_state=42, n_jobs=-1
    )
else:
    clf = GradientBoostingClassifier(
        n_estimators=args.trees, learning_rate=0.1, max_depth=5, random_state=42
    )

print(f"\n   Model    : {clf.__class__.__name__}")
print(f"   Training…")

clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_scores = cross_val_score(clf, X, y_enc, cv=StratifiedKFold(5), scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Save model ────────────────────────────────────────────────────────────────
with open(args.output, "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le}, f)
print(f"\n💾 Model saved → {args.output}")

# ── Plots ─────────────────────────────────────────────────────────────────────
if not args.no_plot:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {clf.__class__.__name__} (Acc: {acc*100:.1f}%)", fontsize=14)
    plt.tight_layout()
    os.makedirs("./docs", exist_ok=True)
    plt.savefig("./docs/confusion_matrix.png", dpi=120)
    plt.show()
    print("📊 Confusion matrix saved → docs/confusion_matrix.png")
