import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_dataset
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 1
RESULTS_DIR = "validation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("adaptive_model_2.keras")

# -----------------------------
# Load Validation Dataset
# -----------------------------
val_ds = load_dataset("val").batch(BATCH_SIZE)

# -----------------------------
# Collect True Labels & Probabilities
# -----------------------------
y_true_list = []
y_prob_list = []

for images, labels in val_ds:
    probs = model.predict(images, verbose=0).flatten()

    y_true_list.extend(labels.numpy())
    y_prob_list.extend(probs)

y_true = np.array(y_true_list)
y_prob = np.array(y_prob_list)

# -----------------------------
# Compute ROC and Best Threshold
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc_score = roc_auc_score(y_true, y_prob)

# Youden's J statistic
J = tpr - fpr
best_idx = np.argmax(J)
best_threshold = thresholds[best_idx]

print("\nBest Threshold (Youden's J):", best_threshold)
print("Validation AUC:", auc_score)

# -----------------------------
# Apply Best Threshold
# -----------------------------
y_pred = (y_prob > best_threshold).astype(int)

# -----------------------------
# Validation Metrics
# -----------------------------
print("\nValidation Metrics Using Best Threshold")
print("----------------------------------------")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing"]))

# -----------------------------
# Plot ROC Curve
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

# Mark best threshold point
plt.scatter(fpr[best_idx], tpr[best_idx], marker="o", label="Best Threshold")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(os.path.join(RESULTS_DIR, "validation_roc_curve.png"))
plt.close()

print("\n✔ Use this threshold in your API:")
print("THRESHOLD =", float(best_threshold))