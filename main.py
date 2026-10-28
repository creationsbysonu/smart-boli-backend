import io
import os
import magic
import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

MODEL_PATH = os.path.join(os.path.dirname(__file__), "nepali_stutter_model.pkl")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
N_MFCC = 13
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
ALLOWED_MIME_TYPES = {"audio/webm", "audio/mp4", "audio/wav", "audio/mpeg", "audio/ogg"}

app = FastAPI(title="Smart Boli - Nepali Stutter Detection Backend")

# Security: Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "smart-boli-backend.onrender.com",
        "localhost",
        "127.0.0.1",
    ],
)

# Security: Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Restrictive CORS - only allow known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stutterdetector.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)


# Security: Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "microphone=(), camera=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictResponse(BaseModel):
    label: str
    color: str
    fluent: bool
    confidence: float | None = None
    chunk_seconds: float


class ErrorResponse(BaseModel):
    detail: str


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
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/warmup")
def warmup():
    try:
        _get_model()
        return {"status": "warmed", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    # Security: Validate file type
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    # Security: Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio file")

    # Security: Validate MIME type using file magic bytes
    try:
        mime = magic.from_buffer(contents, mime=True)
        if mime not in ALLOWED_MIME_TYPES and not mime.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Invalid audio format detected: {mime}",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If magic detection fails, proceed with content_type check

    try:
        model = _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    try:
        audio, sr = librosa.load(io.BytesIO(contents), sr=None, mono=True)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not decode audio: {e}")

    if len(audio) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio")

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
    color = "#10b981" if fluent else "#ef4444"

    return PredictResponse(
        label=label,
        color=color,
        fluent=fluent,
        confidence=confidence,
        chunk_seconds=CHUNK_SECONDS,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
