from fastapi import FastAPI
from pydantic import BaseModel
from src.backend.scraper import scrape_html
from src.backend.predictor import predict
from src.backend.htmltotensor import html_to_tensor, html_to_fixed_tensor
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from src.backend.predictor import load_model
import os
import json
from datetime import datetime
from pathlib import Path
import tempfile
from datetime import timedelta

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

class ReportRequest(BaseModel):
    url: str
    model_name: str
  

# Counter files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COUNTER_FILE = PROJECT_ROOT / "model_performance_counters.json"
# Keep the name aligned with the frontend/user expectation
LOG_FILE = PROJECT_ROOT / "feedback_logs.json"

def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)

def initialize_counters():
    """Initialize counter file if it doesn't exist"""
    if not COUNTER_FILE.exists():
        _atomic_write_json(COUNTER_FILE, {"false_alarms": 0, "missed_phishing": 0})

def increment_counter(counter_name):
    """Increment a counter value"""
    initialize_counters()
    with open(COUNTER_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if counter_name in data:
        data[counter_name] += 1
    else:
        data[counter_name] = 1
    _atomic_write_json(COUNTER_FILE, data)
    return data

def _read_log_entries():
    if not LOG_FILE.exists():
        _atomic_write_json(LOG_FILE, [])
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def _is_duplicate_feedback(existing_entries, new_entry, dedupe_window_seconds: int = 10) -> bool:
    """
    Prevent accidental double-click duplicates.
    Duplicate definition: same (url, type, model) as the most recent matching entry,
    and its timestamp is within a short window.
    """
    try:
        new_ts = datetime.fromisoformat(new_entry["timestamp"])
    except Exception:
        return False

    # scan backwards so we only look at the most recent relevant entry
    for prev in reversed(existing_entries):
        if (
            prev.get("url") == new_entry.get("url")
            and prev.get("type") == new_entry.get("type")
            and prev.get("model") == new_entry.get("model")
        ):
            try:
                prev_ts = datetime.fromisoformat(prev.get("timestamp", ""))
            except Exception:
                return False
            return abs((new_ts - prev_ts).total_seconds()) <= dedupe_window_seconds
    return False

# Log Feedback
def log_feedback(url, report_type, model):
    entry = {
        "url": url,
        "type": report_type,
        "model": model,
        "timestamp": datetime.utcnow().isoformat()
    }

    data = _read_log_entries()

    # If the user double-clicks quickly, don't create a duplicate entry.
    if _is_duplicate_feedback(data, entry, dedupe_window_seconds=10):
        return None

    data.append(entry)

    _atomic_write_json(LOG_FILE, data)
    return entry


@app.on_event("startup")
def startup_event():
    initialize_counters()
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

@app.post("/report_false_alarm")
def report_false_alarm(request: ReportRequest):
    entry = log_feedback(request.url, "false_alarm", request.model_name)
    if entry is None:
        return {"message": "duplicate_ignored", "status": "success"}
    counters = increment_counter("false_alarms")
    return {"message": "reported", "status": "success", "counters": counters}

@app.post("/report_missed_phishing")
def report_missed_phishing(request: ReportRequest):
    entry = log_feedback(request.url, "missed_phishing", request.model_name)
    if entry is None:
        return {"message": "duplicate_ignored", "status": "success"}
    counters = increment_counter("missed_phishing")
    return {"message": "reported", "status": "success", "counters": counters}

@app.get("/model_performance")
def get_model_performance():
    """Get current model performance counters"""
    initialize_counters()
    with open(COUNTER_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
