from fastapi import FastAPI
from pydantic import BaseModel
from src.backend.scraper import scrape_html
from src.backend.predictor import predict
from src.backend.htmltotensor import html_to_tensor, html_to_fixed_tensor
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from src.backend.predictor import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str
    model_name: str



@app.on_event("startup")
def startup_event():
    load_model("AdaptiveCNN")
    load_model("BaselineCNN")

@app.post("/predict")
def predict_url(request: URLRequest):

    html = scrape_html(request.url)

    # choose preprocessing
    if request.model_name == "AdaptiveCNN":
        image_array = html_to_tensor(html)

    elif request.model_name == "BaselineCNN":
        image_array = html_to_fixed_tensor(html)

    else:
        return {"error": "Invalid model selected"}

    image_array = tf.expand_dims(image_array, axis=0)

    label, confidence = predict(image_array, request.model_name)

    return {
        "url": request.url,
        "model_used": request.model_name,
        "prediction": label,
        "confidence": round(confidence, 4)
    }