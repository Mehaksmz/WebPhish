import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

SIZE = 224
IMG_SIZE = (SIZE, SIZE)
BATCH_SIZE = 32
THRESHOLD = 0.5
DATA_DIR = "/home/mehak/image-dataset"

model = tf.keras.models.load_model("resnet50_final.keras")


test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",      # MUST match training
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names  # ['legitimate', 'phishing']


y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = (y_prob > THRESHOLD).astype(int).flatten()


print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing"]))


titles_options = [
    ("Confusion Matrix (Counts)", None),
    ("Normalized Confusion Matrix", "true")
]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, normalize=normalize),
        display_labels=["Legitimate", "Phishing"]
    )
    disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.show()

false_positives = []
false_negatives = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = (probs > THRESHOLD).astype(int).flatten()

    for img, true, pred in zip(images, labels.numpy(), preds):
        if true == 0 and pred == 1:
            false_positives.append(img)
        elif true == 1 and pred == 0:
            false_negatives.append(img)

print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")


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
    plt.show()


show_images(false_positives, "False Positives (Legitimate → Predicted Phishing)")
show_images(false_negatives, "False Negatives (Phishing → Predicted Legitimate)")


images, labels = next(iter(test_ds))
pred_probs = model.predict(images)
pred_labels = (pred_probs > THRESHOLD).astype(int).flatten()

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap="gray")

    true_label = class_names[int(labels[i])]
    pred_label = class_names[int(pred_labels[i])]

    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    plt.axis("off")

plt.suptitle("Test Images: True Label vs Predicted Label", fontsize=16)
plt.tight_layout()
plt.show()
