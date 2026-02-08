import tensorflow as tf
from preprocess_optimized import load_dataset, load_dataset_with_resize

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(e)

# Choose your loading strategy:
# Option 1: Bucketing (variable sizes, memory efficient)
train_ds = load_dataset("train", use_bucketing=True)
val_ds = load_dataset("val", use_bucketing=True)

# Option 2: Fixed resize (if you prefer consistent shapes)
# Uncomment below to use fixed-size approach instead
# train_ds = load_dataset_with_resize("train", target_height=2048, batch_size=16)
# val_ds = load_dataset_with_resize("val", target_height=2048, batch_size=16)

# Build model with adaptive input shape
model = tf.keras.Sequential([
    # Input layer that accepts variable sizes
    tf.keras.layers.InputLayer(input_shape=(None, None, 1)),
    
    tf.keras.layers.Rescaling(1./255),
    
    # Conv layers work with any input size
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    
    # Global pooling handles variable spatial dimensions
    tf.keras.layers.GlobalAveragePooling2D(),
    
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
print("\nModel Summary:")
model.build(input_shape=(None, None, None, 1))
model.summary()

# Training with memory-efficient settings
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            monitor="val_loss",
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        # Monitor memory during training
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"\nEpoch {epoch+1} - "
                f"Train Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
        )
    ],
    verbose=1
)

model.save("adaptive_model.keras")
print("\nModel saved successfully!")

# Save training history
import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)
print("Training history saved to training_history.csv")
