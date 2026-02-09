import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "/home/mehak/image-dataset"


# ================= DATA =================
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

    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(base.input, output)

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


# ================= FINE-TUNE =================
def fine_tune(model, unfreeze=20):

    base_model = model.layers[0]  # the pretrained CNN

    for layer in base_model.layers[-unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


# ================= TRAIN =================
def train(name):

    print(f"\n==== Training {name.upper()} ====\n")

    model = build_model(name)

    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        "resnet50_best.keras",
        monitor="val_loss",
        save_best_only=True
    )

    # Stage 1
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS,
              callbacks=[early, checkpoint])

    # Stage 2
    model = fine_tune(model)
    model.fit(train_data, validation_data=val_data, epochs=10,
              callbacks=[early, checkpoint])

    # Save final
    # os.makedirs("models", exist_ok=True)
    model.save("resnet50_final.keras")

 


# ================= RUN =================
# for m in ["resnet", "vgg", "inception"]:
train("resnet")
