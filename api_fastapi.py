"""
api_fastapi.py — FastAPI Backend Service untuk Deteksi Phishing URL
UAS Pembelajaran Mesin | A11.2024.15574 — Mayolus Gavin

Jalankan dengan: uvicorn api_fastapi:app --reload
"""

import os
import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn

# ── Load model ──
MODEL_DIR = "models"

model = joblib.load(os.path.join(MODEL_DIR, "best_phishing_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.joblib"))

with open(os.path.join(MODEL_DIR, "best_model_info.json")) as f:
    import json
    model_info = json.load(f)

# ── FastAPI ──
app = FastAPI(
    title="Phishing URL Detection API",
    description="Deteksi URL phishing menggunakan SVM optimized",
    version="1.0.0"
)

# ── Schema ──
class URLInput(BaseModel):
    url: str

class FeaturesInput(BaseModel):
    URLLength: float = 50
    DomainLength: float = 15
    IsDomainIP: float = 0
    URLSimilarityIndex: float = 100.0
    CharContinuationRate: float = 0.5
    TLDLegitimateProb: float = 0.5
    URLCharProb: float = 0.05
    TLDLength: float = 3
    NoOfSubDomain: float = 1
    HasObfuscation: float = 0
    NoOfObfuscatedChar: float = 0
    ObfuscationRatio: float = 0.0
    NoOfLettersInURL: float = 30
    LetterRatioInURL: float = 0.7
    NoOfDegitsInURL: float = 5
    DegitRatioInURL: float = 0.1
    NoOfEqualsInURL: float = 0
    NoOfQMarkInURL: float = 0
    NoOfAmpersandInURL: float = 0
    NoOfOtherSpecialCharsInURL: float = 3
    SpacialCharRatioInURL: float = 0.1
    IsHTTPS: float = 1

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_legitimate: float
    probability_phishing: float
    model: str
    model_accuracy: float

# ── Feature extractor ──
def extract_url_features(url_str: str) -> dict:
    if not url_str.startswith(("http://", "https://")):
        url_parse = "http://" + url_str
    else:
        url_parse = url_str
    parsed = urlparse(url_parse)
    domain = parsed.netloc
    parts = domain.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    letters = sum(1 for c in url_str if c.isalpha())
    digits = sum(1 for c in url_str if c.isdigit())
    special = sum(1 for c in url_str if not c.isalnum() and c not in ["/", ":", ".", "-", "_"])

    return {
        "URLLength": len(url_str),
        "DomainLength": len(domain),
        "IsDomainIP": 1 if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", domain) else 0,
        "URLSimilarityIndex": 100.0,
        "CharContinuationRate": 0.5,
        "TLDLegitimateProb": 0.5,
        "URLCharProb": 0.05,
        "TLDLength": len(tld),
        "NoOfSubDomain": max(0, len(parts) - 2) if len(parts) > 1 else 0,
        "HasObfuscation": 1 if "%20" in url_str or "@" in url_str else 0,
        "NoOfObfuscatedChar": url_str.count("%") + url_str.count("@"),
        "ObfuscationRatio": (url_str.count("%") + url_str.count("@")) / max(len(url_str), 1),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / max(len(url_str), 1),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / max(len(url_str), 1),
        "NoOfEqualsInURL": url_str.count("="),
        "NoOfQMarkInURL": url_str.count("?"),
        "NoOfAmpersandInURL": url_str.count("&"),
        "NoOfOtherSpecialCharsInURL": special,
        "SpacialCharRatioInURL": special / max(len(url_str), 1),
        "IsHTTPS": 1 if url_str.lower().startswith("https") else 0,
    }

# ── Predict function ──
def predict(features: dict) -> dict:
    df = pd.DataFrame([features])
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0.0)
    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    return {
        "prediction": int(pred),
        "label": "PHISHING" if pred == 1 else "LEGITIMATE",
        "probability_legitimate": round(float(proba[0]), 4),
        "probability_phishing": round(float(proba[1]), 4),
        "model": str(model_info.get("model_name", "SVM")),
        "model_accuracy": model_info.get("accuracy", 0),
    }

# ── Endpoints ──
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Phishing URL Detection API",
        "docs": "/docs",
        "model": model_info.get("model_name"),
        "accuracy": model_info.get("accuracy"),
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict/url", response_model=PredictionResponse, tags=["Prediction"])
def predict_from_url(input: URLInput):
    """Prediksi dari URL string mentah — fitur diekstrak otomatis."""
    try:
        features = extract_url_features(input.url)
        result = predict(features)
        result["url"] = input.url
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/features", response_model=PredictionResponse, tags=["Prediction"])
def predict_from_features(input: FeaturesInput):
    """Prediksi dari fitur manual (langsung input 22 fitur)."""
    try:
        features = input.model_dump()
        result = predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info", tags=["Info"])
def model_info_endpoint():
    """Info model yang sedang digunakan."""
    return model_info

# ── Run ──
if __name__ == "__main__":
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
