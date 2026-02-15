import tensorflow as tf
from preprocess import load_dataset

train_ds = load_dataset("train2").batch(1)
val_ds = load_dataset("val2").batch(1)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="adaptive_weights.h5",   # where to save
    monitor="val_loss",                  # same metric as EarlyStopping
    save_best_only=True,                 # only best epoch
    save_weights_only=True,              # save weights, not full model
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=6,
    monitor="val_loss",
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# optional: save full best model after training finishes
model.save("adaptive_model_2.keras")

