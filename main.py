import io
import os
import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "nepali_stutter_model.pkl")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
N_MFCC = 13

app = FastAPI(title="Smart Boli - Nepali Stutter Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


class PredictResponse(BaseModel):
    label: str
    color: str
    fluent: bool
    confidence: float | None = None
    chunk_seconds: float


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return features


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        audio, sr = librosa.load(io.BytesIO(contents), sr=None, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    features = extract_features(audio, sr)

    pred = int(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        except Exception:
            confidence = None

    fluent = pred == 0
    label = "FLUENT" if fluent else "STUTTERED"
    color = "#10b981" if fluent else "#ef4444"  # green / red

    return PredictResponse(
        label=label,
        color=color,
        fluent=fluent,
        confidence=confidence,
        chunk_seconds=CHUNK_SECONDS,
    )
