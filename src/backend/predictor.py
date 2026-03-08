# predictor.py
import tensorflow as tf

models = {}

MODEL_CONFIG = {
    "AdaptiveCNN": {
        "path": "models/adaptive_model_2.keras",
        "threshold": 0.39844608306884766
    },
    "BaselineCNN": {
        "path": "models/cnn_phishing_model.keras",
        "threshold": 0.5
    }
}

def load_model(model_name):
    if model_name not in models:
        config = MODEL_CONFIG[model_name]
        models[model_name] = tf.keras.models.load_model(config["path"])
        models[model_name].steps_per_execution = 1

    return models[model_name]


def predict(image_array, model_name):
    model = load_model(model_name)

    threshold = MODEL_CONFIG[model_name]["threshold"]

    prediction = model.predict(image_array)[0][0]
    confidence = float(prediction)

    label = "Phishing" if confidence > threshold else "Legitimate"

    return label, confidence