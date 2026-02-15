import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# ⚡ Configuration
# -----------------------------
THRESHOLD = 0.5
BATCH_SIZE = 1  # variable image heights
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("adaptive_model_2.keras")

# -----------------------------
# Load test dataset
# -----------------------------
test_ds = load_dataset("test").batch(BATCH_SIZE)

# -----------------------------
# Prepare true labels, predictions, probabilities
# -----------------------------
y_true_list = []
y_pred_list = []
y_prob_list = []

false_positives = []
false_negatives = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0).flatten()
    preds = (probs > THRESHOLD).astype(int)

    y_true_list.extend(labels.numpy())
    y_pred_list.extend(preds)
    y_prob_list.extend(probs)

    # Store misclassified examples
    for img, true, pred in zip(images, labels.numpy(), preds):
        if true == 0 and pred == 1:
            false_positives.append(img)
        elif true == 1 and pred == 0:
            false_negatives.append(img)

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
y_prob = np.array(y_prob_list)

# -----------------------------
# Metrics
# -----------------------------
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing"]))

# -----------------------------
# Confusion matrices
# -----------------------------
# titles_options = [
#     ("Confusion Matrix (Counts)", None),
#     ("Normalized Confusion Matrix", "true")
# ]

# for title, normalize in titles_options:
#     disp = ConfusionMatrixDisplay(
#         confusion_matrix=confusion_matrix(y_true, y_pred, normalize=normalize),
#         display_labels=["Legitimate", "Phishing"]
#     )
#     disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
#     plt.title(title)
#     plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}.png"))
#     plt.close()

# -----------------------------
# ROC Curve and AUC (FIXED)
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc_score = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

# -----------------------------
# Visualization of misclassified images
# -----------------------------
def show_images(images, title, max_images=10):
    if len(images) == 0:
        print(f"No images to display for {title}")
        return

    plt.figure(figsize=(15, 6))
    for i, img in enumerate(images[:max_images]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.numpy().squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}.png"))
    plt.close()

show_images(false_positives, "False Positives (Legitimate → Predicted Phishing)")
show_images(false_negatives, "False Negatives (Phishing → Predicted Legitimate)")

# -----------------------------
# Preview first 10 test images
# -----------------------------
preview_images = []
preview_labels = []

for images, labels in test_ds.take(10):
    preview_images.append(images[0])
    preview_labels.append(labels[0])

plt.figure(figsize=(15, 6))
for i in range(len(preview_images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(preview_images[i].numpy().squeeze(), cmap="gray")
    plt.title(f"True: {int(preview_labels[i])}\nPred: ?", fontsize=10)
    plt.axis("off")

plt.suptitle("Test Images Preview", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "test_preview.png"))
plt.close()
