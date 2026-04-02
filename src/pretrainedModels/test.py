import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# -----------------------------
# PARAMETERS
# -----------------------------
SIZE = 224
IMG_SIZE = (SIZE, SIZE)
BATCH_SIZE = 32
THRESHOLD = 0.5
DATA_DIR = "preprocess"
RESULTS_DIR = "results/inception"

os.makedirs(RESULTS_DIR, exist_ok=True)

# # -----------------------------
# # BUILD VGG MODEL
# # -----------------------------
# def build_model():
#     base = VGG19(weights="imagenet", include_top=False,
#                  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

#     for layer in base.layers:
#         layer.trainable = False

#     x = GlobalAveragePooling2D()(base.output)
#     x = Dense(256, activation="relu")(x)
#     x = Dropout(0.5)(x)
#     output = Dense(1, activation="sigmoid")(x)

#     model = Model(base.input, output)

#     model.compile(
#         optimizer=Adam(1e-4),
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )

#     return model

# # -----------------------------
# # LOAD MODEL WEIGHTS
# # -----------------------------
# model = build_model()
# model.load_weights("vgg_weights.h5")

model = tf.keras.models.load_model("resnet50_best.keras")


# -----------------------------
# LOAD TEST DATASET
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds).flatten()
y_pred = (y_prob > THRESHOLD).astype(int)

# -----------------------------
# METRICS
# -----------------------------
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# CONFUSION MATRICES
# -----------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_true, y_pred),
    display_labels=class_names
)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix (Counts)inception")
plt.savefig(os.path.join(RESULTS_DIR, "inceptionV3.png"))
plt.close()

# # -----------------------------
# # ROC CURVE + AUC
# # -----------------------------
# fpr, tpr, thresholds = roc_curve(y_true, y_prob)
# auc_score = roc_auc_score(y_true, y_prob)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
# plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.savefig(os.path.join(RESULTS_DIR, "roc_curvevgg.png"))
# plt.close()

# # -----------------------------
# # FALSE POSITIVES & FALSE NEGATIVES
# # -----------------------------
# false_positives = []
# false_negatives = []

# for images, labels in test_ds:
#     probs = model.predict(images, verbose=0).flatten()
#     preds = (probs > THRESHOLD).astype(int)

#     for img, true, pred in zip(images, labels.numpy(), preds):
#         if true == 0 and pred == 1:
#             false_positives.append(img)
#         elif true == 1 and pred == 0:
#             false_negatives.append(img)

# print(f"False Positives VGG: {len(false_positives)}")
# print(f"False Negatives VGG: {len(false_negatives)}")

# def show_images(images, title, max_images=10):
#     if len(images) == 0:
#         print(f"No images to display for {title}")
#         return

#     plt.figure(figsize=(15, 6))
#     for i, img in enumerate(images[:max_images]):
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(img.numpy())
#         plt.axis("off")

#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}.png"))
#     plt.close()

# show_images(false_positives, "False Positives (Legitimate → Predicted Phishing)")
# show_images(false_negatives, "False Negatives (Phishing → Predicted Legitimate)")

# # -----------------------------
# # PREVIEW FIRST 10 TEST IMAGES
# # -----------------------------
# preview_images = []
# preview_labels = []

# for images, labels in test_ds.unbatch().take(10):
#     preview_images.append(images)
#     preview_labels.append(labels)

# plt.figure(figsize=(15, 6))
# for i in range(len(preview_images)):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(preview_images[i].numpy())
#     true_label = class_names[int(preview_labels[i])]
#     pred_label = class_names[int((model.predict(preview_images[i][tf.newaxis, ...]).flatten() > THRESHOLD))]
#     plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
#     plt.axis("off")

# plt.suptitle("Test Images Preview", fontsize=16)
# plt.tight_layout()
# plt.savefig(os.path.join(RESULTS_DIR, "test_previewvgg.png"))
# plt.close()
