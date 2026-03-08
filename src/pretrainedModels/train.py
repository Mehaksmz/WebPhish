import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from sklearn.utils import class_weight

IMG_SIZE = 299
BATCH_SIZE = 32
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 10
DATA_DIR = "preprocess"


# ================= PREPROCESSING =================
def get_preprocess_function(name):
    if name == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif name == "vgg":
        from tensorflow.keras.applications.vgg19 import preprocess_input
    elif name == "inception":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    else:
        raise ValueError("Invalid model name")
    return preprocess_input


# ================= DATA =================
def get_data_generators(preprocess_input):
    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb"
    )

    val_data = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb"
    )

    return train_data, val_data


# ================= MODEL =================
def build_model(name):

    if name == "resnet":
        base = ResNet50(weights="imagenet", include_top=False,
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif name == "vgg":
        base = VGG19(weights="imagenet", include_top=False,
                     input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif name == "inception":
        base = InceptionV3(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Invalid model")

    # Freeze base layers
    for layer in base.layers:
        layer.trainable = False

    # Classification head
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(base.input, output)

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model, base


# ================= FINE-TUNE =================
def fine_tune(model, base_model, unfreeze=120):

    for layer in base_model.layers[-unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(5e-6),  # lower LR for stability
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


# ================= TRAIN =================
def train(name):

    print(f"\n==== Training {name.upper()} ====\n")

    preprocess_input = get_preprocess_function(name)
    train_data, val_data = get_data_generators(preprocess_input)

    # -------- CLASS WEIGHTS (KEY FIX) --------
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weights = dict(enumerate(weights))
    print("Class weights:", class_weights)

    # -------- MODEL --------
    model, base_model = build_model(name)

    os.makedirs("models", exist_ok=True)

    early = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        f"models/{name}_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    # ===== Stage 1 =====
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_STAGE1,
        callbacks=[early, checkpoint, reduce_lr],
        class_weight=class_weights
    )

    # ===== Stage 2 (fine-tune deeper) =====
    model = fine_tune(model, base_model)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_STAGE2,
        callbacks=[early, checkpoint, reduce_lr],
        class_weight=class_weights
    )

    # Save final
    model.save(f"models/{name}_final.keras")
    print(f"\n✅ {name.upper()} training complete.\n")


# ================= RUN =================
train("inception")   # change to resnet / inception if needed
